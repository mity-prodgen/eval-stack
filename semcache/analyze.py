#!/usr/bin/env python3
"""Metrics, bootstrap CIs, figures and RESULTS.md from semcache/results/{scores,decisions}_full.jsonl.

Definitions (n with every rate; CIs are seeded 95% bootstrap over pairs, 2,000 resamples):
  served          a lookup returns the cached answer. A0: cosine >= t (t = 0.90 unless stated).
  wrong-hit rate  traps (label "different") served / traps.
  control hit     controls (label "same") served / controls - the benefit of the cache.
  A1              A0 with the highest threshold t whose control hits are >= 80% of A0's (at 0.90), chosen on the
                  OTHER fold of a seeded 2-fold split and applied to this fold, so no pair picks its own threshold.
  A2              A0 and every slot matches (semcache/slots.py).
  A3-<decider>    A0 and P(yes) >= 0.5 for "Do these two queries need the same answer?". A decision that is not
                  status ok fails open (served), and is counted under harness checks.
  blocked / lost  for a fix vs A0 on the same pairs: traps A0 served that the fix does not (blocked) and controls A0
                  served that the fix does not (lost), as counts and as shares of A0's wrong hits / control hits.
  cost / 1,000    one new-query embedding per lookup (actual billed cost per text; gemini-embedding-2 reports no
                  usage, so it is estimated) plus, for A3, one decision per lookup that passes A0, at this set's A0
                  pass rate (actual mean cost per decision). Local models: $0 in API fees.

The main set (pairs.jsonl) and PAWS are reported separately. Source groups (authored, gpt-6.1-sol,
claude-sonnet-5-5) are reported separately where the author could matter.

    python -m semcache.analyze [--smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from harness.metrics import auc, bootstrap_ci

from .build_pairs import SLOT_TYPES, source_group
from .build_paws import load_paws
from .common import (A1_KEEP, A3_PASS, DECIDER_MODELS, DECIDERS, DEFAULT_COSINE, DEFAULTS, EMBEDDER, EMBEDDERS,
                     FIGURES, PKG, RESULTS, RESULTS_MD, SEED, THRESHOLDS)
from .estimate import estimate_embed
from .run import load_pairs, smoke_pairs
from .slots import guard

N_BOOT = 2000
GROUPS = ("authored", "gpt-6.1-sol", "claude-sonnet-5-5")
ARMS = ("A0", "A1", "A2", *[f"A3-{k}" for k in DECIDERS])


# ------------------------------------------------------------------------------------------- loading
def load(stem: str):
    pairs = smoke_pairs(load_pairs()) if stem == "smoke" else load_pairs()
    paws = [] if stem == "smoke" else load_paws()
    scores: dict[tuple[str, str], float] = {}
    for line in (RESULTS / f"scores_{stem}.jsonl").open():
        r = json.loads(line)
        scores[(r["embedder"], r["pair_id"])] = r["cosine"]
    decisions: dict[tuple[str, str], dict] = {}
    path = RESULTS / f"decisions_{stem}.jsonl"
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            decisions[(r["decider"], r["pair_id"])] = r
    return pairs, paws, scores, decisions


def _ci(vals, seed):
    return bootstrap_ci(vals, n_boot=N_BOOT, seed=seed) if vals else None


def present(scores, pairs) -> list:
    """Embedding models with a score for every pair (a model still waiting on quota is left out, and said so)."""
    return [e for e in EMBEDDERS if all((e.name, p["id"]) in scores for p in pairs)]


def rate(vals: list[float], seed: int) -> dict:
    return {"n": len(vals), "k": int(sum(vals)), "ci": _ci(vals, seed)}


def fold(pid: str) -> int:
    return int(hashlib.sha256(f"{SEED}|{pid}".encode()).hexdigest(), 16) % 2


# ------------------------------------------------------------------------------------------- arms
class Arms:
    """served(arm, model, pair) for every arm; A1 thresholds chosen out of fold."""

    def __init__(self, pairs, scores, decisions, guards):
        self.scores, self.decisions, self.guards = scores, decisions, guards
        self.a1_t: dict[tuple[str, int], float] = {}
        controls = [p["id"] for p in pairs if p["label"] == "same"]
        for e in EMBEDDERS:
            for f in (0, 1):
                other = [c for c in controls if fold(c) != f and (e.name, c) in scores]
                base = sum(scores[(e.name, c)] >= DEFAULT_COSINE for c in other)
                ok = [t for t in THRESHOLDS if t >= DEFAULT_COSINE
                      and sum(scores[(e.name, c)] >= t for c in other) >= A1_KEEP * base]
                self.a1_t[(e.name, f)] = max(ok) if ok else DEFAULT_COSINE
        self.fail_open = Counter()

    def a0(self, model, pid, t=DEFAULT_COSINE) -> bool:
        return self.scores[(model, pid)] >= t

    def served(self, arm, model, pid) -> bool:
        if arm == "A0":
            return self.a0(model, pid)
        if arm == "A1":
            return self.a0(model, pid, self.a1_t[(model, fold(pid))])
        if not self.a0(model, pid):
            return False
        if arm == "A2":
            return self.guards[pid]
        k = arm.split("-", 1)[1]
        d = self.decisions.get((k, pid))
        if d is None or d["status"] != "ok" or d["p_yes"] is None:
            self.fail_open[(k, d["status"] if d else "missing")] += 1
            return True
        return d["p_yes"] >= A3_PASS


# ------------------------------------------------------------------------------------------- metrics
def arm_metrics(arms: Arms, pairs: list[dict], seed: int) -> dict:
    traps = [p["id"] for p in pairs if p["label"] == "different"]
    ctrls = [p["id"] for p in pairs if p["label"] == "same"]
    out = {}
    for e in EMBEDDERS:
        m = e.name
        if not all((m, p) in arms.scores for p in traps + ctrls):
            continue
        a0_w = [p for p in traps if arms.served("A0", m, p)]
        a0_c = [p for p in ctrls if arms.served("A0", m, p)]
        for k, arm in enumerate(ARMS):
            if arm.startswith("A3") and not arms.decisions:
                continue
            w = [float(arms.served(arm, m, p)) for p in traps]
            c = [float(arms.served(arm, m, p)) for p in ctrls]
            blocked = [float(not arms.served(arm, m, p)) for p in a0_w]
            lost = [float(not arms.served(arm, m, p)) for p in a0_c]
            out[f"{m}|{arm}"] = {"wrong_hit": rate(w, seed + k), "control_hit": rate(c, seed + 10 + k),
                                 "blocked": rate(blocked, seed + 20 + k), "lost": rate(lost, seed + 30 + k)}
    return out


def curves(scores, pairs) -> dict:
    """Point estimates over the threshold grid (no CIs: 50 thresholds x 8 models)."""
    traps = [p["id"] for p in pairs if p["label"] == "different"]
    ctrls = [p["id"] for p in pairs if p["label"] == "same"]
    out = {}
    for e in EMBEDDERS:
        if (e.name, traps[0]) not in scores:
            continue
        out[e.name] = [(t, statistics.mean(scores[(e.name, p)] >= t for p in traps),
                        statistics.mean(scores[(e.name, p)] >= t for p in ctrls)) for t in THRESHOLDS]
    return out


def at_defaults(scores, pairs, seed) -> dict:
    traps = [p["id"] for p in pairs if p["label"] == "different"]
    ctrls = [p["id"] for p in pairs if p["label"] == "same"]
    out = {}
    for e in present(scores, pairs):
        for d in DEFAULTS:
            t = -1.0 if d.cosine is None else d.cosine
            out[f"{e.name}|{d.library}"] = {
                "cosine": d.cosine,
                "wrong_hit": rate([float(scores[(e.name, p)] >= t) for p in traps], seed),
                "control_hit": rate([float(scores[(e.name, p)] >= t) for p in ctrls], seed + 1)}
    return out


def by_slot(scores, pairs, seed, t=DEFAULT_COSINE) -> dict:
    out = {}
    for e in present(scores, pairs):
        for s in SLOT_TYPES:
            ids = [p["id"] for p in pairs if p["label"] == "different" and p["slot_type"] == s]
            out[f"{e.name}|{s}"] = rate([float(scores[(e.name, p)] >= t) for p in ids], seed)
    return out


def by_group(arms: Arms, pairs, seed) -> dict:
    out = {}
    for g in GROUPS:
        sub = [p for p in pairs if source_group(p) == g]
        for key, v in arm_metrics(arms, sub, seed).items():
            out[f"{g}|{key}"] = v
    return out


def separation(scores, pairs, seed, n_boot: int = 1000) -> dict:
    """AUC = P(cosine of a random control > cosine of a random trap); 0.5 = cosine can't tell them apart.
    CI: bootstrap resampling traps and controls separately."""
    import random
    traps = [p["id"] for p in pairs if p["label"] == "different"]
    ctrls = [p["id"] for p in pairs if p["label"] == "same"]
    out = {}
    for e in present(scores, pairs):
        t = [scores[(e.name, p)] for p in traps]
        c = [scores[(e.name, p)] for p in ctrls]
        rng = random.Random(seed)
        boots = sorted(auc([rng.choice(c) for _ in c], [rng.choice(t) for _ in t]) for _ in range(n_boot))
        out[e.name] = {"auc": auc(c, t), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot)],
                       "n_traps": len(t), "n_controls": len(c)}
    return out


def hypotheses(scores, pairs, seed) -> dict:
    """Direction and negation traps: share with cosine >= 0.95, and the median, per model."""
    out = {}
    for s in ("direction", "negation"):
        ids = [p["id"] for p in pairs if p["label"] == "different" and p["slot_type"] == s]
        for e in present(scores, pairs):
            cs = [scores[(e.name, p)] for p in ids]
            out[f"{e.name}|{s}"] = {"share_ge_095": rate([float(c >= 0.95) for c in cs], seed),
                                    "median": statistics.median(cs)}
    return out


def guard_alone(pairs, guards, seed) -> dict:
    """The slot guard on every pair, independent of any embedding: catch rate on traps, false blocks on controls."""
    out = {}
    for g in (*GROUPS, "all"):
        sub = [p for p in pairs if g == "all" or source_group(p) == g]
        for s in SLOT_TYPES:
            ids = [p["id"] for p in sub if p["label"] == "different" and p["slot_type"] == s]
            out[f"{g}|catch|{s}"] = rate([float(not guards[p]) for p in ids], seed)
        out[f"{g}|catch|all"] = rate([float(not guards[p["id"]]) for p in sub if p["label"] == "different"], seed)
        for s in sorted({p["slot_type"] for p in sub if p["label"] == "same"}):
            ids = [p["id"] for p in sub if p["label"] == "same" and p["slot_type"] == s]
            out[f"{g}|false_block|{s}"] = rate([float(not guards[p]) for p in ids], seed + 1)
        out[f"{g}|false_block|all"] = rate([float(not guards[p["id"]]) for p in sub if p["label"] == "same"], seed + 1)
    return out


def decider_alone(pairs, decisions, seed) -> dict:
    """Each decider on the pairs it was asked about (those passing A0 at 0.90 on some model)."""
    out = {}
    for k in DECIDERS:
        for g in (*GROUPS, "all"):
            sub = [p for p in pairs if g == "all" or source_group(p) == g]
            for label, name in (("different", "accepts_trap"), ("same", "rejects_control")):
                vals = []
                for p in sub:
                    d = decisions.get((k, p["id"]))
                    if p["label"] == label and d and d["status"] == "ok":
                        yes = d["p_yes"] >= A3_PASS
                        vals.append(float(yes if label == "different" else not yes))
                out[f"{k}|{g}|{name}"] = rate(vals, seed)
    return out


def costs(decisions, scores, pairs) -> dict:
    """USD per 1,000 lookups per model x arm (see module docstring)."""
    usage = defaultdict(lambda: [0, 0.0])
    path = RESULTS / "embed_usage.jsonl"
    if path.exists():
        for line in path.open():
            u = json.loads(line)
            usage[u["embedder"]][0] += u["n_texts"]
            usage[u["embedder"]][1] += u["cost_usd"] or 0.0
    per_text, estimated = {}, set()
    texts = [p["new"] for p in pairs]
    for e in EMBEDDERS:
        n, usd = usage[e.name]
        if e.provider == "local":
            per_text[e.name] = 0.0
        elif usd > 0:
            per_text[e.name] = usd / n
        else:
            per_text[e.name] = estimate_embed(e, texts).usd / len(texts)
            estimated.add(e.name)
    per_decision = {}
    for k in DECIDERS:
        cs = [d["cost_usd"] for (kk, _), d in decisions.items() if kk == k and d.get("cost_usd") is not None]
        per_decision[k] = statistics.mean(cs) if cs else None
    out = {}
    for e in present(scores, pairs):
        pass_rate = statistics.mean(scores[(e.name, p["id"])] >= DEFAULT_COSINE for p in pairs)
        for arm in ARMS:
            usd = per_text[e.name]
            if arm.startswith("A3"):
                k = arm.split("-", 1)[1]
                if per_decision.get(k) is None:
                    continue
                usd += pass_rate * per_decision[k]
            out[f"{e.name}|{arm}"] = {"usd_per_1000": 1000 * usd, "embed_estimated": e.name in estimated,
                                      "a0_pass_rate": pass_rate}
    return {"per_lookup": out, "per_decision": per_decision}


# ------------------------------------------------------------------------------------------- output helpers
def pct(m: dict | None) -> str:
    if not m or not m["ci"]:
        return "- (n=0)"
    p, lo, hi = m["ci"]
    return f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] (n={m['n']})"


def frac(m: dict) -> str:
    return f"{m['k']}/{m['n']}" + (f" ({100 * m['ci'][0]:.0f}%)" if m["ci"] else "")


# ------------------------------------------------------------------------------------------- figures
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE = "#2a78d6"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
DEFAULT_COLUMNS = [(0.90, "RedisVL"), (0.80, "LangChain Redis, both packages"), (0.70, "Cassandra"),
                   (0.60, "GPTCache")]
DEFAULT_MARKERS = {0.90: ("o", "RedisVL 0.90"), 0.80: ("s", "LangChain Redis 0.80"), 0.70: ("^", "Cassandra 0.70"),
                   0.60: ("D", "GPTCache 0.60")}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(cv: dict, slot: dict, defaults: dict, out_dir: Path, suffix: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    models = [e.name for e in EMBEDDERS if e.name in cv]

    # 1. wrong-hit vs control-hit, one panel per model, library defaults marked
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, m in zip(axes.flat, models):
        _style(ax)
        pts = cv[m]
        ax.plot([c for _, _, c in pts], [w for _, w, _ in pts], color=BLUE, linewidth=2, solid_capstyle="round")
        for t, (mk, _) in DEFAULT_MARKERS.items():
            d = next(v for k, v in defaults.items() if k.startswith(f"{m}|") and v["cosine"] == t)
            ax.plot(d["control_hit"]["ci"][0], d["wrong_hit"]["ci"][0], mk, markersize=7, color=INK,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        for t, w, c in pts:
            if t in (0.95, 0.99):
                ax.annotate(f"{t:.2f}", (c, w), xytext=(4, -2), textcoords="offset points", fontsize=6.5, color=INK2)
        ax.set_title(f"{m} ({EMBEDDER[m].model_id.split('/')[-1]})", fontsize=8, color=INK, loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)  # a model still waiting on quota gets no empty panel
    for ax in axes[1]:
        ax.set_xlabel("control hit rate (benefit)", fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("wrong-hit rate (traps served)", fontsize=8, color=INK2)
    handles = [Line2D([], [], color=BLUE, linewidth=2, label="cosine threshold 0.50 -> 0.99")] + [
        Line2D([], [], marker=mk, linestyle="", color=INK, markersize=7, label=f"default: {lab}")
        for mk, lab in DEFAULT_MARKERS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Wrong hits vs control hits as the threshold rises (main set; down-right is better)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    p = out_dir / f"tradeoff{suffix}.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # 2. wrong-hit rate by slot type at 0.90 (heatmap, single-hue sequential)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    grid = [[slot[f"{m}|{s}"]["ci"][0] if slot[f"{m}|{s}"]["ci"] else float("nan") for s in SLOT_TYPES] for m in models]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7.5,
                    color=INK if v < 0.55 else "#ffffff")
    ax.set_xticks(range(len(SLOT_TYPES)), [s.replace("_", "/") for s in SLOT_TYPES], fontsize=8, color=INK)
    ax.set_yticks(range(len(models)), models, fontsize=8, color=INK)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(SLOT_TYPES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    n = slot[f"{models[0]}|{SLOT_TYPES[0]}"]["n"]
    ax.set_title(f"Wrong-hit rate by slot type at cosine >= 0.90 (RedisVL default), main set, ~{n} traps per cell",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    p = out_dir / f"slot_types{suffix}.png"
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------- RESULTS.md
LIMITATIONS = [
    "Synthetic queries. Every pair is constructed: templates, pairs written by the author (Claude Opus 5.5), and "
    "pairs written by gpt-6.1-sol and Claude Sonnet 5.5 (80 of them edits of real BANKING77 queries). Real cache "
    "traffic has a different mix: most lookups are neither near-identical traps nor clean paraphrases, so these "
    "rates are conditional on a lookup being one of the two, not production hit rates.",
    "The authored pairs and the slot guard share one author, and the guard's alias tables were written knowing "
    "the slot types. The guard was committed before the pairs existed and not changed after, but its numbers on "
    "the authored group are likely optimistic; the gpt-6.1-sol and claude-sonnet-5-5 groups and PAWS are the "
    "check, reported separately.",
    "Labels are the generators' own plus one human review of 30 pairs (29/30 confirmed; Wilson 95% CI 83-99%). "
    "At that n, a few percent of labels can be wrong; generator-assigned slot types are noisier than labels "
    "(e.g. some BANKING77 'direction' traps are action swaps).",
    "The main tables compare each lookup with one stored query. The full-cache simulation (721 stored queries, "
    "nearest entry served) changed wrong-hit rates by at most 2 points for 7 of 8 models (9 for gemini-embedding-2): a "
    "trap's nearest neighbour is almost always its own one-detail-different partner. That store is small and spread "
    "across topics; a production cache full of near-variants of popular queries is not modelled.",
    "Cosine on unit vectors for every model. Library defaults are converted to cosine (see the defaults table); "
    "GPTCache's own default ONNX embedding is not normalized and OpenSearch's default effectively has no "
    "threshold, so their rows describe the threshold applied to these embeddings, not those libraries end to end.",
    "One request setting per embedding model (query mode on both sides where a model has one; Gemini's "
    "documented similarity prefix). Other modes, dimensions or prompts can move the cosine distribution.",
    "A3 asks each decider once per pair, only for pairs that pass A0 at 0.90 on at least one model, at default "
    "sampling. Run-to-run variance of the deciders is not measured here (post 1's noise baseline measured it "
    "for judging: Jev and Luna changed 0-2.5% of verdicts between runs).",
    "A1 picks its threshold out of fold on the same pair distribution; on a different traffic mix the 80%-keep "
    "threshold would differ.",
    "Bootstrap CIs resample pairs; pairs from the same template or generator request are not independent, so "
    "the intervals are somewhat too narrow.",
    "gemini-embedding-2 responses carry no token usage, so its cost is the dry-run estimate (which ran 5-22% "
    "above actual billing for the other hosted models).",
]

HARNESS_BUGS = [
    "gemini-embedding-2 returns ONE aggregated embedding when `contents` is a list of strings; each text now goes "
    "in its own `Content`, and every client asserts one vector per input.",
    "An inline `genai.Client()` was garbage-collected mid-call, closing its HTTP client (found in the smoke test).",
    "A3 cost estimate was ~2x low: Jev and Luna bill a near-constant per-request overhead (+257..267 / +129..130 "
    "tokens) that post 1's long-passage factor missed. Recalibrated from the smoke test.",
    "`generate_pairs` first wrote a partial frozen set when every Sonnet request hit the account's API usage "
    "limit; it now refuses to freeze unless every request succeeded.",
    "`build_pairs --refreeze` redrew review.csv and would have erased the human review; it no longer does.",
    "Post 1 Claude batch for Haiku 4.5 (noise re-run 2) ended with all 360 requests errored on low credit; errors "
    "are never cached, so the re-run retried them at no double cost.",
]


def write_md(M: dict, figs: list[Path], path: Path, smoke: bool) -> None:
    models = M["models"]
    L = [f"# Results: how often does a semantic cache serve the wrong stored answer?{' (SMOKE)' if smoke else ''}", "",
         "Regenerated by `semcache/analyze.py`; not hand-edited (findings come from `semcache/findings.md`).", ""]
    n = M["n"]
    L += [f"Main set: {n['traps']} traps (need a different answer) and {n['controls']} controls (need the same "
          f"answer), {n['by_group']}. PAWS: {n['paws_traps']} / {n['paws_controls']}. {len(models)} embedding models"
          + (f" ({', '.join(M['models_missing'])} missing: see harness checks)" if M["models_missing"] else "")
          + ", deciders "
          + ", ".join(f"{k} (`{DECIDER_MODELS[k]}`)" for k in DECIDERS) + ".", ""]
    findings = PKG / "findings.md"
    if findings.exists() and not smoke:
        L += ["## Top findings", "", findings.read_text().strip(), ""]

    L += ["## Library defaults applied to each model (main set)", "",
          "Wrong-hit rate / control hit rate at each library's default threshold, converted to cosine. "
          "OpenSearch's default passes every pair (100% / 100%) and SingleStore's 0.20 almost every pair.", "",
          "| Model | " + " | ".join(f"{lab} ({c:.2f})" for c, lab in DEFAULT_COLUMNS) + " |",
          "|---|" + "---|" * len(DEFAULT_COLUMNS)]
    for m in models:
        cells = []
        for c, _ in DEFAULT_COLUMNS:
            d = next(d for d in DEFAULTS if d.cosine == c)
            v = M["defaults"][f"{m}|{d.library}"]
            cells.append(f"{100 * v['wrong_hit']['ci'][0]:.0f}% / {100 * v['control_hit']['ci'][0]:.0f}%")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", f"n = {n['traps']} traps, {n['controls']} controls per cell; 95% CIs in `results/metrics_full.json`.", ""]

    L += ["## A0 at the RedisVL default (cosine >= 0.90)", "",
          "AUC = chance that a random control (same answer) scores higher than a random trap (different answer); "
          "0.5 means cosine cannot tell them apart, below 0.5 means traps score higher.", "",
          "| Model | wrong-hit rate | control hit rate | AUC main set [95% CI] | AUC PAWS [95% CI] |",
          "|---|---|---|---|---|"]
    for m in models:
        a = M["arms"][f"{m}|A0"]
        u, up = M["auc"][m], M["auc_paws"].get(m)
        paws_auc = f"{up['auc']:.2f} [{up['lo']:.2f}, {up['hi']:.2f}]" if up else "-"
        L.append(f"| {m} (`{EMBEDDER[m].model_id}`) | {pct(a['wrong_hit'])} | {pct(a['control_hit'])} | "
                 f"{u['auc']:.2f} [{u['lo']:.2f}, {u['hi']:.2f}] | {paws_auc} |")
    L += ["", "## Wrong-hit rate by slot type at 0.90", "", "| Model | " + " | ".join(SLOT_TYPES) + " |",
          "|---|" + "---|" * len(SLOT_TYPES)]
    for m in models:
        L.append(f"| {m} | " + " | ".join(pct(M["slot"][f"{m}|{s}"]) for s in SLOT_TYPES) + " |")

    L += ["", "## Fixes vs A0, same pairs", "",
          "blocked = A0's wrong hits the fix stops; lost = A0's control hits the fix gives up. Both always reported.",
          "", "| Model | Arm | wrong-hit | control hit | wrong hits blocked | control hits lost | $ / 1,000 lookups |",
          "|---|---|---|---|---|---|---|"]
    for m in models:
        for arm in ARMS:
            a = M["arms"].get(f"{m}|{arm}")
            if not a:
                continue
            c = M["costs"]["per_lookup"].get(f"{m}|{arm}")
            cost = f"${c['usd_per_1000']:.4f}" + ("*" if c["embed_estimated"] else "") if c else "-"
            L.append(f"| {m} | {arm} | {pct(a['wrong_hit'])} | {pct(a['control_hit'])} | {pct(a['blocked'])} | "
                     f"{pct(a['lost'])} | {cost} |")
    L += ["", "A1 thresholds chosen out of fold (fold 0 / fold 1): " + ", ".join(
        f"{m} {M['a1_t'][f'{m}|0']:.2f}/{M['a1_t'][f'{m}|1']:.2f}" for m in models if f"{m}|0" in M["a1_t"]) + ".",
          "\\* gemini-embedding-2 reports no usage; its embedding cost is the estimate. Local models: $0 API cost "
          "(compute not counted). A3 cost assumes this set's A0 pass rate.", ""]

    L += ["## Does the author matter? Fixes by source group (A0 at 0.90)", "",
          "Same metrics on each source group. The slot guard (A2) and the authored pairs share one author.", "",
          "| Model | Arm | " + " | ".join(f"{g}: blocked / lost" for g in GROUPS) + " |", "|---|---|" + "---|" * len(GROUPS)]
    for m in models:
        for arm in ARMS[1:]:
            cells = []
            for g in GROUPS:
                a = M["groups"].get(f"{g}|{m}|{arm}")
                cells.append(f"{pct(a['blocked'])} / {pct(a['lost'])}" if a else "-")
            L.append(f"| {m} | {arm} | " + " | ".join(cells) + " |")

    L += ["", "## Slot guard alone (no embedding): catch rate on traps, false blocks on controls", "",
          "| Group | " + " | ".join(SLOT_TYPES) + " | all traps | all controls (false block) |",
          "|---|" + "---|" * (len(SLOT_TYPES) + 2)]
    for g in (*GROUPS, "all"):
        L.append(f"| {g} | " + " | ".join(pct(M["guard"].get(f"{g}|catch|{s}")) for s in SLOT_TYPES) +
                 f" | {pct(M['guard'][f'{g}|catch|all'])} | {pct(M['guard'][f'{g}|false_block|all'])} |")
    L += ["", "False blocks by control type (all groups): " + "; ".join(
        f"{k.split('|')[2]} {pct(v)}" for k, v in M["guard"].items() if k.startswith("all|false_block|")
        and not k.endswith("|all")) + ".", ""]

    if M["deciders"]:
        L += ["## Deciders alone (pairs that pass A0 at 0.90 on some model)", "",
              "| Decider | Group | accepts a trap (P(yes) >= 0.5) | rejects a control |", "|---|---|---|---|"]
        for k in DECIDERS:
            for g in (*GROUPS, "all"):
                L.append(f"| {k} | {g} | {pct(M['deciders'][f'{k}|{g}|accepts_trap'])} | "
                         f"{pct(M['deciders'][f'{k}|{g}|rejects_control'])} |")
        L.append("")

    L += ["## Hypotheses", "",
          "H1: direction and negation traps score above 0.95 on most models. Share of those traps with cosine >= 0.95 "
          "(median cosine):", "", "| Model | direction | negation |", "|---|---|---|"]
    for m in models:
        h = M["hyp"]
        L.append(f"| {m} | " + " | ".join(f"{pct(h[f'{m}|{s}']['share_ge_095'])} (median {h[f'{m}|{s}']['median']:.3f})"
                                          for s in ("direction", "negation")) + " |")
    L += ["", "H2: the slot guard blocks most traps but loses hits on aliases: see the slot-guard tables above "
          "(catch rate per slot type; false blocks by control type).", ""]

    if M.get("paws"):
        L += ["## PAWS (external check: word-swap pairs, 150 / 150)", "",
              "| Model | Arm | wrong-hit | control hit | blocked | lost |", "|---|---|---|---|---|---|"]
        for m in models:
            for arm in ARMS:
                a = M["paws"].get(f"{m}|{arm}")
                if a:
                    L.append(f"| {m} | {arm} | {pct(a['wrong_hit'])} | {pct(a['control_hit'])} | {pct(a['blocked'])} | "
                             f"{pct(a['lost'])} |")
        L.append("")

    sim_path = RESULTS / "simulation.json"
    if sim_path.exists() and not smoke:
        S = json.loads(sim_path.read_text())
        store = next(iter(S.values()))["store_size"]
        L += ["## Full-cache simulation (`semcache/simulate.py`, offline)", "",
              f"Every pair's new query is looked up against all {store} distinct stored queries; the nearest is served "
              "if it clears the threshold. A hit on the pair's own partner carries the pair's label; a hit on any "
              "other stored query is unlabelled, so the trap wrong-hit rate is a range: low = partner hits only, "
              "high = every unlabelled hit counted as wrong. An exact-text match counts as correct. Many unlabelled "
              "hits read as correct matches (e.g. a Business Class trap finding another Business Class query), so "
              "the true rate sits near the low end.", "",
              "| Model | Arm | t | traps served, single pair | traps wrong, full cache (low - high) | controls served "
              "correctly, single pair | full cache |", "|---|---|---|---|---|---|---|"]
        for m in models:
            for arm in ("A0", "A2"):
                for t in ("0.90", "0.95"):
                    v = S.get(f"{m}|{arm}|{t}")
                    if v:
                        L.append(f"| {m} | {arm} | {t} | {100 * v['trap_single_pair']:.1f}% | "
                                 f"{100 * v['trap_wrong_low']:.1f}% - {100 * v['trap_wrong_high']:.1f}% | "
                                 f"{100 * v['control_single_pair']:.1f}% | {100 * v['control_correct']:.1f}% |")
        L.append("")

    L += ["## Harness checks", "", "```json", json.dumps(M["checks"], indent=1), "```", "",
          "## Harness bugs found and fixed", ""] + [f"- {b}" for b in HARNESS_BUGS]
    L += ["", "## Figures", ""] + [f"![{p.stem}]({p.relative_to(path.parent)})" for p in figs]
    L += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS] + [""]
    path.write_text("\n".join(L))


# ------------------------------------------------------------------------------------------- main
def compute(stem: str) -> tuple[dict, dict]:
    pairs, paws, scores, decisions = load(stem)
    guards = {p["id"]: guard(p["cached"], p["new"])[0] for p in pairs + paws}
    arms = Arms(pairs, scores, decisions, guards)
    M = {
        "n": {"traps": sum(p["label"] == "different" for p in pairs), "controls": sum(p["label"] == "same" for p in pairs),
              "by_group": ", ".join(f"{g} {sum(source_group(p) == g for p in pairs)}" for g in GROUPS),
              "paws_traps": sum(p["label"] == "different" for p in paws),
              "paws_controls": sum(p["label"] == "same" for p in paws)},
        "models": [e.name for e in present(scores, pairs + paws)],
        "models_missing": [e.name for e in EMBEDDERS if e not in present(scores, pairs + paws)],
        "arms": arm_metrics(arms, pairs, SEED),
        "a1_t": {f"{m}|{f}": t for (m, f), t in arms.a1_t.items()},
        "defaults": at_defaults(scores, pairs, SEED + 100),
        "slot": by_slot(scores, pairs, SEED + 200),
        "groups": by_group(arms, pairs, SEED + 300),
        "guard": guard_alone(pairs, guards, SEED + 400),
        "deciders": decider_alone(pairs, decisions, SEED + 500) if decisions else {},
        "hyp": hypotheses(scores, pairs, SEED + 600),
        "auc": separation(scores, pairs, SEED + 800),
        "auc_paws": separation(scores, paws, SEED + 900) if paws else {},
        "costs": costs(decisions, scores, pairs),
        "paws": arm_metrics(arms, paws, SEED + 700) if paws else {},
    }
    asked = {pid for (_, pid) in decisions}
    need = {p["id"] for p in pairs + paws if any(scores.get((e.name, p["id"]), 0) >= DEFAULT_COSINE for e in EMBEDDERS)}
    M["checks"] = {
        "scores_missing": {e.name: sum((e.name, p["id"]) not in scores for p in pairs + paws) for e in EMBEDDERS
                           if any((e.name, p["id"]) not in scores for p in pairs + paws)},
        "decision_status": dict(Counter(f"{d['decider']}|{d['status']}" for d in decisions.values())),
        "a0_passing_pairs_without_decision": len(need - asked) if decisions else None,
        "a3_fail_open": {f"{k}|{s}": v for (k, s), v in arms.fail_open.items()},
        "decider_api_models": dict(Counter(f"{d['decider']}|{d['api_model']}" for d in decisions.values())),
    }
    return M, {"curves": curves(scores, pairs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    stem = "smoke" if args.smoke else "full"
    M, extra = compute(stem)
    fig_dir = RESULTS if args.smoke else FIGURES
    figs = figures(extra["curves"], M["slot"], M["defaults"], fig_dir, "_smoke" if args.smoke else "")
    out = RESULTS / "smoke_RESULTS.md" if args.smoke else RESULTS_MD
    write_md(M, figs, out, args.smoke)
    (RESULTS / f"metrics_{stem}.json").write_text(json.dumps({**M, **extra}, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
