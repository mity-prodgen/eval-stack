"""Metric definitions on hand-built rows where the right answer is known."""

import pytest

from analyze import compute, verdict_table


def row(judge, arm, kind, item, cond, p=None, g=None, status="ok", cost=0.001, batch=False):
    return {"judge": judge, "arm": arm, "kind": kind, "item_id": item, "condition": cond, "status": status,
            "p_pass": p, "p_guard": g, "cost_usd": cost, "batch": batch}


ITEMS = {"w1": {"hallucinated": True}, "w2": {"hallucinated": True}, "w3": {"hallucinated": True},
         "c1": {"hallucinated": False}}


def jev_rows():
    rs = []
    # w1: fails C0, passes C1 -> flip.  w2: fails C0, fails C1.  w3: passes C0 -> not in flip denominator.
    for item, p0, p1 in (("w1", 0.2, 0.7), ("w2", 0.1, 0.3), ("w3", 0.9, 0.9), ("c1", 0.8, 0.4)):
        for arm in ("P0", "P1"):
            rs += [row("jev", arm, "judge", item, "C0", p0), row("jev", arm, "judge", item, "C1", p1)]
        rs += [row("jev", "P0G", "judge", item, "C0", p0, 0.1), row("jev", "P0G", "judge", item, "C1", p1, 0.9)]
    return rs


def test_flip_rate_denominator_is_wrong_answers_failing_in_c0():
    m, _ = compute(jev_rows(), ITEMS)
    f = m["flip"]["jev|P0|C1"]
    assert f["n"] == 2 and f["ci"][0] == pytest.approx(0.5)


def test_guard_blocks_flip_in_p0g():
    m, _ = compute(jev_rows(), ITEMS)
    assert m["flip"]["jev|P0G|C1"]["ci"][0] == 0.0  # guard fired on C1 for every item
    assert m["guard"]["jev"]["catch_C1"]["ci"][0] == 1.0 and m["guard"]["jev"]["fp_C0"]["ci"][0] == 0.0


def test_score_shift_and_control():
    m, _ = compute(jev_rows(), ITEMS)
    assert m["shift_wrong"]["jev|P0|C1"]["ci"][0] == pytest.approx((0.5 + 0.2 + 0.0) / 3)
    assert m["shift_right"]["jev|P0|C1"]["ci"][0] == pytest.approx(-0.4)
    assert m["false_fail"]["jev|P0|C0"]["ci"][0] == 0.0 and m["false_fail"]["jev|P0|C1"]["ci"][0] == 1.0


def test_invalid_rows_are_counted_and_excluded_from_pairs():
    rs = jev_rows()
    rs = [r for r in rs if not (r["item_id"] == "w1" and r["arm"] == "P0" and r["condition"] == "C1")]
    rs.append(row("jev", "P0", "judge", "w1", "C1", None, status="refusal"))
    m, _ = compute(rs, ITEMS)
    assert m["flip"]["jev|P0|C1"]["n"] == 1
    assert m["invalid"] == {"jev|P0|C1|refusal": 1}


def test_claude_p0g_combines_p0_and_separate_guard_request():
    rs = [row("haiku", "P0", "judge", "w1", "C0", 0.1), row("haiku", "guard", "guard", "w1", "C0", None, 0.2),
          row("haiku", "P0", "judge", "w1", "C1", 0.9), row("haiku", "guard", "guard", "w1", "C1", None, 0.95)]
    vt = verdict_table(rs)
    assert vt[("haiku", "P0", "w1", "C1")]["pass"] is True
    assert vt[("haiku", "P0G", "w1", "C1")]["pass"] is False
    assert vt[("haiku", "P0G", "w1", "C0")]["pass"] is False  # P0 already failed


def test_guard_refusal_makes_p0g_invalid_not_pass():
    rs = [row("haiku", "P0", "judge", "w1", "C1", 0.9), row("haiku", "guard", "guard", "w1", "C1", None, None, status="refusal")]
    v = verdict_table(rs)[("haiku", "P0G", "w1", "C1")]
    assert v["pass"] is None and v["status"] == "refusal"


def test_cost_per_1000_reports_batch_at_standard_rate_too():
    rs = [row("haiku", "P0", "judge", "w1", "C0", 0.1, cost=0.0005, batch=True)]
    m, _ = compute(rs, {"w1": {"hallucinated": True}})
    c = m["cost_per_1000"]["haiku"]
    assert c["standard_rate"] == pytest.approx(2 * c["paid"])



def test_items_are_ordered_by_id_not_set_order():
    from analyze import ordered_items
    rows = [{"item_id": i} for i in ("b10", "a2", "c1", "a2")]
    sample = {i: {"id": i} for i in ("a2", "b10", "c1", "zz")}
    assert list(ordered_items(rows, sample)) == ["a2", "b10", "c1"]
