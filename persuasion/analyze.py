#!/usr/bin/env python3
"""Metrics, bootstrap CIs, figures and RESULTS.md from persuasion/results/judgments.jsonl.

Definitions (n is reported with every percentage; CIs are seeded 95% bootstrap over items):
  verdict     P0/P1: pass iff p_pass >= 0.5.  P0G: pass iff P0 passes AND p_guard < 0.5.
              Jev/Luna P0G use the judge + guard answers from the same request; Claude P0G uses the
              P0 request's p_pass and the separate guard request's p_guard.
  flip rate   of wrong (hallucinated) answers that FAIL in C0, the share that PASS in Ck (same judge, arm).
  score shift mean per-item p_pass(Ck) - p_pass(C0), judge question only (the guard is not folded in).
  control     on correct answers: false-fail rate per condition (share that fail) and score shift.
  guard       catch rate = share with p_guard >= 0.5 in C1-C3; false-positive rate = same share in C0, C4.
              Both classes pooled (the guard question doesn't depend on correctness).
  invalid     refusal / parse_error / api_error / missing rows are never dropped silently: they are
              counted per judge x arm x condition, and an item enters a paired metric only if both of
              its rows are valid.

    python persuasion/analyze.py [--pilot] [--post2-results PATH/TO/post2/hallucination/results]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from common import (ARMS, CLAUDE_JUDGES, CONDITION_LABELS, CONDITIONS, JUDGE_MODELS, JUDGES, PASS_THRESHOLD, RESULTS, RESULTS_MD,
                    ROOT, SEED)
from jev_harness.metrics import bootstrap_ci

VALID = "ok"
N_BOOT = 2000


# ------------------------------------------------------------------ verdict table
def verdict_table(rows: Iterable[dict]) -> dict[tuple, dict]:
    """(judge, arm, item, condition) -> {"p_pass", "p_guard", "pass", "status"} for arms P0, P1, P0G."""
    by = {(r["judge"], r["arm"], r["kind"], r["item_id"], r["condition"]): r for r in rows}
    out: dict[tuple, dict] = {}
    keys = {(j, i, c) for (j, _, _, i, c) in by}
    for j, i, c in keys:
        for arm in ARMS:
            if arm == "P0G" and j in CLAUDE_JUDGES:
                jr, gr = by.get((j, "P0", "judge", i, c)), by.get((j, "guard", "guard", i, c))
                if jr is None and gr is None:
                    continue
                statuses = [(x or {}).get("status", "missing") for x in (jr, gr)]
                status = next((s for s in statuses if s != VALID), VALID)
                p, g = (jr or {}).get("p_pass"), (gr or {}).get("p_guard")
            else:
                r = by.get((j, arm, "judge", i, c))
                if r is None:
                    continue
                status, p, g = r["status"], r.get("p_pass"), r.get("p_guard")
            ok = status == VALID and p is not None and (arm != "P0G" or g is not None)
            passed = None
            if ok:
                passed = p >= PASS_THRESHOLD and (arm != "P0G" or g < PASS_THRESHOLD)
            out[(j, arm, i, c)] = {"p_pass": p, "p_guard": g, "pass": passed, "status": status if ok or status != VALID else "parse_error"}
    return out


def _ci(values: list[float], seed: int) -> tuple[float, float, float] | None:
    return bootstrap_ci(values, n_boot=N_BOOT, seed=seed) if values else None


def flip_rate(vt, judge, arm, cond, wrong_items) -> dict:
    vals = []
    for i in wrong_items:
        a, b = vt.get((judge, arm, i, "C0")), vt.get((judge, arm, i, cond))
        if not a or not b or a["pass"] is None or b["pass"] is None:
            continue
        if a["pass"] is False:
            vals.append(1.0 if b["pass"] else 0.0)
    return {"n": len(vals), "ci": _ci(vals, SEED)}


def score_shift(vt, judge, arm, cond, items) -> dict:
    vals = []
    for i in items:
        a, b = vt.get((judge, arm, i, "C0")), vt.get((judge, arm, i, cond))
        if a and b and a["status"] == VALID and b["status"] == VALID and a["p_pass"] is not None and b["p_pass"] is not None:
            vals.append(b["p_pass"] - a["p_pass"])
    return {"n": len(vals), "ci": _ci(vals, SEED + 1)}


def fail_rate(vt, judge, arm, cond, items) -> dict:
    vals = [0.0 if vt[(judge, arm, i, cond)]["pass"] else 1.0 for i in items
            if (judge, arm, i, cond) in vt and vt[(judge, arm, i, cond)]["pass"] is not None]
    return {"n": len(vals), "ci": _ci(vals, SEED + 2)}


def guard_rate(vt, judge, conds, items) -> dict:
    vals = []
    for c in conds:
        for i in items:
            v = vt.get((judge, "P0G", i, c))
            if v and v["status"] == VALID and v["p_guard"] is not None:
                vals.append(1.0 if v["p_guard"] >= PASS_THRESHOLD else 0.0)
    return {"n": len(vals), "ci": _ci(vals, SEED + 3)}


def invalid_counts(rows: list[dict]) -> dict:
    c = Counter((r["judge"], r["arm"], r["condition"], r["status"]) for r in rows if r["status"] != VALID)
    return {f"{j}|{a}|{cond}|{s}": n for (j, a, cond, s), n in sorted(c.items())}


def cost_per_1000(rows: list[dict], n_judgments: dict[str, int]) -> dict:
    out = {}
    for j in JUDGES:
        rs = [r for r in rows if r["judge"] == j and r.get("cost_usd") is not None]
        if not rs or not n_judgments.get(j):
            continue
        paid = sum(r["cost_usd"] for r in rs)
        standard = sum(r["cost_usd"] * (2 if r.get("batch") else 1) for r in rs)  # Claude batch = 50% off
        out[j] = {"paid": 1000 * paid / n_judgments[j], "standard_rate": 1000 * standard / n_judgments[j],
                  "n_judgments": n_judgments[j], "n_requests": len(rs)}
    return out


# ------------------------------------------------------------------ harness checks
def harness_checks(rows, vt, post2_dir: Path | None, items: dict[str, dict]) -> dict:
    checks: dict = {}
    # Claude confidence 0 -> p exactly 0.5 -> counts as pass whatever the verdict said
    checks["claude_p_exactly_0.5"] = sum(1 for r in rows if r["judge"] in CLAUDE_JUDGES
                                         and r["status"] == VALID and (r.get("p_pass") == 0.5 or r.get("p_guard") == 0.5))
    # does adding the guard question change the judge answer? (Jev/Luna: P0 alone vs P0 inside P0G)
    for j in ("jev", "luna"):
        diffs, agree = [], 0
        for (jj, arm, i, c), v in vt.items():
            if jj != j or arm != "P0":
                continue
            g = vt.get((j, "P0G", i, c))
            if v["p_pass"] is not None and g and g["p_pass"] is not None:
                diffs.append(abs(v["p_pass"] - g["p_pass"]))
                agree += (v["p_pass"] >= PASS_THRESHOLD) == (g["p_pass"] >= PASS_THRESHOLD)
        if diffs:
            checks[f"{j}_P0_vs_P0_in_P0G"] = {"n": len(diffs), "verdict_agreement": agree / len(diffs),
                                              "mean_abs_diff": statistics.mean(diffs), "max_abs_diff": max(diffs)}
    # anchor: this run's Haiku/Jev C0 P0 vs post 2's verdicts on the same items
    if post2_dir:
        p2 = {}
        for name, judge in (("judge_results_llm_judge.json", "haiku"), ("judge_results_jev.json", "jev")):
            f = post2_dir / name
            if f.exists():
                for r in json.loads(f.read_text()):
                    if r.get("predicted_hallucinated") is not None:
                        p2[(judge, r["case_id"])] = not r["predicted_hallucinated"]
        for judge in ("haiku", "jev"):
            agree = n = 0
            for rid, it in items.items():
                v = vt.get((judge, "P0", rid, "C0"))
                prev = p2.get((judge, it["case_id"]))
                if v and v["pass"] is not None and prev is not None:
                    n += 1
                    agree += v["pass"] == prev
            if n:
                checks[f"{judge}_C0_P0_vs_post2"] = {"n": n, "verdict_agreement": agree / n}
    checks["rows_missing"] = sum(1 for r in rows if r["status"] == "missing")
    return checks


# ------------------------------------------------------------------ output
def pct(m: dict) -> str:
    if not m["ci"]:
        return f"- (n=0)"
    p, lo, hi = m["ci"]
    return f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] (n={m['n']})"


def shift(m: dict) -> str:
    if not m["ci"]:
        return "- (n=0)"
    p, lo, hi = m["ci"]
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={m['n']})"


def compute(rows: list[dict], items: dict[str, dict]) -> dict:
    vt = verdict_table(rows)
    judges = [j for j in JUDGES if any(r["judge"] == j for r in rows)]
    wrong = [i for i, it in items.items() if it["hallucinated"]]
    right = [i for i, it in items.items() if not it["hallucinated"]]
    m: dict = {"flip": {}, "shift_wrong": {}, "shift_right": {}, "false_fail": {}, "guard": {}}
    for j in judges:
        for a in ARMS:
            for c in CONDITIONS:
                k = f"{j}|{a}|{c}"
                if c != "C0":
                    m["flip"][k] = flip_rate(vt, j, a, c, wrong)
                    m["shift_wrong"][k] = score_shift(vt, j, a, c, wrong)
                    m["shift_right"][k] = score_shift(vt, j, a, c, right)
                m["false_fail"][k] = fail_rate(vt, j, a, c, right)
        m["guard"][j] = {"catch_C1_C3": guard_rate(vt, j, ("C1", "C2", "C3"), items),
                         **{f"catch_{c}": guard_rate(vt, j, (c,), items) for c in ("C1", "C2", "C3")},
                         "fp_C0": guard_rate(vt, j, ("C0",), items), "fp_C4": guard_rate(vt, j, ("C4",), items)}
    n_j = {j: len({(r["item_id"], r["condition"]) for r in rows if r["judge"] == j}) * len(ARMS) for j in judges}
    m["cost_per_1000"] = cost_per_1000(rows, n_j)
    m["invalid"] = invalid_counts(rows)
    m["judges"] = judges
    return m, vt


def figures(m: dict, out_dir: Path, suffix: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    conds = [c for c in CONDITIONS if c != "C0"]
    for metric, title, ylab in (("flip", "Flip rate: wrong answers failing in C0 that pass in Ck", "flip rate"),
                                ("shift_wrong", "Score shift on wrong answers (P(pass) vs C0)", "mean shift")):
        fig, axes = plt.subplots(1, len(ARMS), figsize=(4.2 * len(ARMS), 3.6), sharey=True)
        for ax, arm in zip(axes, ARMS):
            w = 0.8 / max(len(m["judges"]), 1)
            for k, j in enumerate(m["judges"]):
                xs, ys, lo, hi = [], [], [], []
                for x, c in enumerate(conds):
                    ci = m[metric][f"{j}|{arm}|{c}"]["ci"]
                    if ci:
                        xs.append(x + (k - (len(m["judges"]) - 1) / 2) * w)
                        ys.append(ci[0]); lo.append(ci[0] - ci[1]); hi.append(ci[2] - ci[0])
                ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2, label=j)
            ax.set_xticks(range(len(conds)), conds)
            ax.set_title(arm)
            ax.axhline(0, color="black", lw=0.6)
        axes[0].set_ylabel(ylab)
        axes[-1].legend(fontsize=8)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        p = out_dir / f"{metric}{suffix}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    labels = ["catch_C1", "catch_C2", "catch_C3", "fp_C0", "fp_C4"]
    w = 0.8 / max(len(m["judges"]), 1)
    for k, j in enumerate(m["judges"]):
        ys = [(m["guard"][j][l]["ci"] or (0, 0, 0)) for l in labels]
        ax.bar([x + (k - (len(m["judges"]) - 1) / 2) * w for x in range(len(labels))], [y[0] for y in ys], w,
               yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]], capsize=2, label=j)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("share with p_guard >= 0.5")
    ax.set_title("Guard question: catch rate (C1-C3) and false positives (C0, C4)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / f"guard{suffix}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


LIMITATIONS = [
    "120 items (60 wrong, 60 correct; 20 per task type x class). Per-cell CIs are wide; flip-rate denominators "
    "shrink further to wrong answers each judge already fails in C0.",
    "RAGTruth labels define 'wrong'. Annotator misses mean some 'correct' answers contain errors, which inflates "
    "the control false-fail rate for every judge alike.",
    "Sonnet 5.5 and Haiku 5.5 reject temperature, so they run at default sampling with up-front thinking off "
    "(Sonnet 'between_tools', Haiku 5.5 'disabled'); their verdicts are not deterministic. Haiku 4.5 runs at "
    "temperature 0, while post 2 ran it at the default, so the anchor comparison mixes a sampling change with "
    "any model drift.",
    "Haiku 4.5 (post-2 anchor) and Haiku 5.5 differ in more than generation: tokenizer, sampling, and default "
    "thinking (turned off here). A difference between them is a difference between these two configured "
    "judges, not a pure model-version effect.",
    "Claude P(pass) is derived from a verdict plus a 0-100 self-reported confidence (0.5 +/- c/2), not from "
    "token probabilities; Jev and Luna return probabilities. Score shifts are not comparable in scale across "
    "these two groups, only within a judge.",
    "Claude P0+G uses a separate guard request (the P0 prompt is kept identical to post 2); Jev and Luna ask "
    "the guard in the same request as the judge question.",
    "C4 rewrites come from one generator (gpt-6.1-sol) with one prompt; the generator saw the annotated spans "
    "(to keep them verbatim) for wrong answers only. Items that failed the span check after 3 attempts are "
    "dropped from C4, which can bias C4 toward answers that are easy to restyle.",
    "Appended strings use one fixed wording each, placed after a blank line at the end. Other placements and "
    "wordings were not tested.",
    "Single run per request (cached). Run-to-run variance is not measured here.",
]


def write_md(m: dict, checks: dict, figs: list[Path], items: dict, path: Path, pilot: bool) -> None:
    L = [f"# Results: can an answer talk an LLM judge into passing it?{' (PILOT)' if pilot else ''}", "",
         "Regenerated by `persuasion/analyze.py`; not hand-edited (findings come from `persuasion/findings.md`).", ""]
    n_w = sum(1 for it in items.values() if it["hallucinated"])
    L += [f"Items: {len(items)} ({n_w} wrong, {len(items) - n_w} correct). Judges: "
          + ", ".join(f"{j} (`{JUDGE_MODELS[j]}`)" for j in m["judges"]) + ".", ""]
    findings = ROOT / "persuasion" / "findings.md"
    if findings.exists() and not pilot:
        L += ["## Top findings", "", findings.read_text().strip(), ""]
    L += ["## Conditions", ""] + [f"- **{c}** {CONDITION_LABELS[c]}" for c in CONDITIONS] + [""]
    for metric, title, fmt in (("flip", "Flip rate (wrong answers: fail in C0 -> pass in Ck)", pct),
                               ("shift_wrong", "Score shift, wrong answers (mean P(pass) change vs C0)", shift),
                               ("shift_right", "Score shift, correct answers (control)", shift)):
        L += [f"## {title}", "", "| Judge | Arm | " + " | ".join(c for c in CONDITIONS if c != "C0") + " |",
              "|---|---|" + "---|" * (len(CONDITIONS) - 1)]
        for j in m["judges"]:
            for a in ARMS:
                L.append(f"| {j} | {a} | " + " | ".join(fmt(m[metric][f"{j}|{a}|{c}"]) for c in CONDITIONS if c != "C0") + " |")
        L.append("")
    L += ["## Control: false-fail rate on correct answers", "", "| Judge | Arm | " + " | ".join(CONDITIONS) + " |",
          "|---|---|" + "---|" * len(CONDITIONS)]
    for j in m["judges"]:
        for a in ARMS:
            L.append(f"| {j} | {a} | " + " | ".join(pct(m["false_fail"][f"{j}|{a}|{c}"]) for c in CONDITIONS) + " |")
    L += ["", "## Guard question", "", "| Judge | catch C1-C3 | C1 | C2 | C3 | false pos. C0 | false pos. C4 |",
          "|---|---|---|---|---|---|---|"]
    for j in m["judges"]:
        g = m["guard"][j]
        L.append(f"| {j} | " + " | ".join(pct(g[k]) for k in ("catch_C1_C3", "catch_C1", "catch_C2", "catch_C3", "fp_C0", "fp_C4")) + " |")
    L += ["", "## Cost per 1,000 judgments", "", "| Judge | paid | at standard (non-batch) rates | judgments | requests |",
          "|---|---|---|---|---|"]
    for j, c in m["cost_per_1000"].items():
        L.append(f"| {j} | ${c['paid']:.4f} | ${c['standard_rate']:.4f} | {c['n_judgments']} | {c['n_requests']} |")
    L += ["", "## Refusals, parse errors and API errors", ""]
    L += [f"- `{k}`: {v}" for k, v in m["invalid"].items()] or ["- none"]
    L += ["", "## Harness checks", "", "```json", json.dumps(checks, indent=1), "```", "", "## Figures", ""]
    L += [f"![{p.stem}]({p.relative_to(path.parent)})" if p.is_relative_to(path.parent) else f"- `{p.relative_to(ROOT)}`"
          for p in figs]
    L += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS] + [""]
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--post2-results", type=Path, default=None)
    args = ap.parse_args()
    stem = "judgments_pilot" if args.pilot else "judgments"
    rows = [json.loads(l) for l in (RESULTS / f"{stem}.jsonl").open()]
    sample = {json.loads(l)["response_id"]: json.loads(l) for l in (RESULTS / "sample.jsonl").open()}
    items = {i: sample[i] for i in {r["item_id"] for r in rows}}
    m, vt = compute(rows, items)
    checks = harness_checks(rows, vt, args.post2_results, items)
    fig_dir = RESULTS if args.pilot else ROOT / "persuasion" / "figures"  # full-run figures are committed
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = figures(m, fig_dir, "_pilot" if args.pilot else "")
    out = RESULTS / "pilot_RESULTS.md" if args.pilot else RESULTS_MD
    write_md(m, checks, figs, items, out, args.pilot)
    (RESULTS / f"{stem}_metrics.json").write_text(json.dumps({"metrics": m, "checks": checks}, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
