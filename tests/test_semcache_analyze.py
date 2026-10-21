"""semcache analysis on small synthetic sets with known answers: arms, A1 out of fold, blocked/lost, fail-open,
missing models, AUC, costs, and run.py's merge across runs. No network."""

import json

import pytest

from semcache import analyze, run
from semcache.analyze import Arms, arm_metrics, fold, present, separation

M = "minilm"  # any configured embedder name; analysis only needs scores keyed by it


def pair(pid, label, slot="direction", source="handwritten"):
    return {"id": pid, "cached": f"c {pid}", "new": f"n {pid}", "label": label, "slot_type": slot,
            "domain": "airline", "source": source}


def dec(k, pid, p_yes, status="ok"):
    return {"decider": k, "pair_id": pid, "status": status, "p_yes": p_yes, "cost_usd": 1e-5, "api_model": k}


# ---------------------------------------------------------------- arms on a set with known answers
@pytest.fixture
def tiny():
    pairs = [pair("t1", "different"), pair("t2", "different"), pair("t3", "different"),
             pair("c1", "same"), pair("c2", "same")]
    scores = {(M, "t1"): 0.97, (M, "t2"): 0.93, (M, "t3"): 0.50, (M, "c1"): 0.95, (M, "c2"): 0.91}
    guards = {"t1": False, "t2": True, "t3": True, "c1": True, "c2": False}  # guard blocks t1 and c2
    decisions = {}
    for k in ("jev113", "jev114", "luna"):
        decisions.update({(k, "t1"): dec(k, "t1", 0.1), (k, "t2"): dec(k, "t2", 0.8),
                          (k, "c1"): dec(k, "c1", 0.9), (k, "c2"): dec(k, "c2", 0.2)})
    return pairs, scores, decisions, guards


def test_a0_a2_a3_served_and_blocked_lost_counts(tiny):
    pairs, scores, decisions, guards = tiny
    m = arm_metrics(Arms(pairs, scores, decisions, guards), pairs, seed=1)
    a0 = m[f"{M}|A0"]
    assert (a0["wrong_hit"]["k"], a0["wrong_hit"]["n"]) == (2, 3)      # t1, t2 >= 0.90
    assert (a0["control_hit"]["k"], a0["control_hit"]["n"]) == (2, 2)
    a2 = m[f"{M}|A2"]
    assert (a2["blocked"]["k"], a2["blocked"]["n"]) == (1, 2)          # guard stops t1 of A0's {t1, t2}
    assert (a2["lost"]["k"], a2["lost"]["n"]) == (1, 2)                # and gives up c2
    a3 = m[f"{M}|A3-luna"]
    assert a3["blocked"]["k"] == 1 and a3["lost"]["k"] == 1            # P(yes) 0.1 on t1, 0.2 on c2
    assert a3["wrong_hit"]["k"] == 1 and a3["control_hit"]["k"] == 1
    # blocked/lost are always reported against A0's own hits (same pairs)
    assert a0["blocked"]["k"] == 0 and a0["lost"]["k"] == 0


def test_a3_fails_open_and_counts_it(tiny):
    pairs, scores, decisions, guards = tiny
    decisions[("luna", "t1")] = dec("luna", "t1", None, status="parse_error")
    del decisions[("luna", "t2")]
    arms = Arms(pairs, scores, decisions, guards)
    assert arms.served("A3-luna", M, "t1") and arms.served("A3-luna", M, "t2")
    assert arms.fail_open[("luna", "parse_error")] == 1 and arms.fail_open[("luna", "missing")] == 1


def test_a3_threshold_is_inclusive(tiny):
    pairs, scores, decisions, guards = tiny
    decisions[("jev113", "t1")] = dec("jev113", "t1", 0.5)
    assert Arms(pairs, scores, decisions, guards).served("A3-jev113", M, "t1")


# ---------------------------------------------------------------- A1: chosen out of fold
def _a1_set():
    pairs = [pair(f"c{i}", "same") for i in range(40)] + [pair(f"t{i}", "different") for i in range(10)]
    scores = {(M, f"c{i}"): 0.90 + 0.0025 * i for i in range(40)}   # controls spread 0.90 .. 0.9975
    scores.update({(M, f"t{i}"): 0.95 for i in range(10)})
    return pairs, scores


def test_a1_threshold_keeps_80pct_of_other_folds_control_hits():
    pairs, scores = _a1_set()
    arms = Arms(pairs, scores, {}, {})
    ctrls = [p["id"] for p in pairs if p["label"] == "same"]
    for f in (0, 1):
        t = arms.a1_t[(M, f)]
        other = [c for c in ctrls if fold(c) != f]
        base = sum(scores[(M, c)] >= 0.90 for c in other)
        assert t >= 0.90
        assert sum(scores[(M, c)] >= t for c in other) >= 0.8 * base
        assert sum(scores[(M, c)] >= round(t + 0.01, 2) for c in other) < 0.8 * base  # highest such threshold


def test_a1_threshold_ignores_the_pairs_it_is_applied_to():
    pairs, scores = _a1_set()
    t0 = Arms(pairs, scores, {}, {}).a1_t[(M, 0)]
    moved = dict(scores)
    for p in pairs:  # change only fold-0 controls: fold 0's threshold must not move
        if p["label"] == "same" and fold(p["id"]) == 0:
            moved[(M, p["id"])] = 0.999
    assert Arms(pairs, moved, {}, {}).a1_t[(M, 0)] == t0


def test_folds_are_balanced_and_stable():
    ids = [f"p{i:03d}" for i in range(400)]
    share = sum(fold(i) for i in ids) / len(ids)
    assert 0.4 < share < 0.6 and [fold(i) for i in ids] == [fold(i) for i in ids]


# ---------------------------------------------------------------- missing models, AUC
def test_present_leaves_out_a_model_with_missing_scores(tiny):
    pairs, scores, _, _ = tiny
    scores = {**scores, ("bge-m3", "t1"): 0.9}  # bge-m3 scored on one pair only
    assert [e.name for e in present(scores, pairs)] == [M]


def test_separation_auc():
    pairs = [pair("t1", "different"), pair("t2", "different"), pair("c1", "same"), pair("c2", "same")]
    hi_ctrl = {(M, "t1"): 0.1, (M, "t2"): 0.2, (M, "c1"): 0.8, (M, "c2"): 0.9}
    hi_trap = {(M, "t1"): 0.8, (M, "t2"): 0.9, (M, "c1"): 0.1, (M, "c2"): 0.2}
    assert separation(hi_ctrl, pairs, seed=1, n_boot=50)[M]["auc"] == 1.0
    assert separation(hi_trap, pairs, seed=1, n_boot=50)[M]["auc"] == 0.0


# ---------------------------------------------------------------- costs
def test_costs_use_billed_usage_and_a0_pass_rate(tiny, tmp_path, monkeypatch):
    pairs, scores, decisions, _ = tiny
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    scores = {**scores, **{("oai-3-small", p["id"]): s for (m, p_id), s in scores.items() for p in pairs
                           if p["id"] == p_id}}
    (tmp_path / "embed_usage.jsonl").write_text(json.dumps({"embedder": "oai-3-small", "n_texts": 10,
                                                            "cost_usd": 0.001}) + "\n")
    c = analyze.costs(decisions, scores, pairs)
    a0, a3 = c["per_lookup"]["oai-3-small|A0"], c["per_lookup"]["oai-3-small|A3-luna"]
    assert a0["usd_per_1000"] == pytest.approx(1000 * 0.001 / 10)
    assert a0["a0_pass_rate"] == pytest.approx(4 / 5)
    assert a3["usd_per_1000"] == pytest.approx(1000 * (0.001 / 10 + 0.8 * 1e-5))
    assert c["per_lookup"][f"{M}|A0"]["usd_per_1000"] == 0  # local model


# ---------------------------------------------------------------- run.py merges results across runs
def test_run_merges_scores_and_decisions_across_runs(tmp_path):
    path = tmp_path / "scores_full.jsonl"
    rows = run._load_rows(path, ("embedder", "pair_id"))
    assert rows == {}
    rows[("minilm", "p1")] = {"pair_id": "p1", "embedder": "minilm", "cosine": 0.5}
    run._write_rows(path, rows)
    later = run._load_rows(path, ("embedder", "pair_id"))
    later[("gemini-2", "p1")] = {"pair_id": "p1", "embedder": "gemini-2", "cosine": 0.9}
    later[("minilm", "p1")] = {"pair_id": "p1", "embedder": "minilm", "cosine": 0.6}  # a re-run replaces
    run._write_rows(path, later)
    final = run._load_rows(path, ("embedder", "pair_id"))
    assert set(final) == {("minilm", "p1"), ("gemini-2", "p1")} and final[("minilm", "p1")]["cosine"] == 0.6
