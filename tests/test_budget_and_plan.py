import json
import sys

import pytest

import run
from common import RESULTS
from estimate import BudgetExceeded, check_budget, estimate_call
from judges import BUILDERS, Call

needs_sample = pytest.mark.skipif(not (RESULTS / "sample.jsonl").exists(), reason="run build_sample.py first")


def test_check_budget():
    check_budget(4.99, 5.0)
    with pytest.raises(BudgetExceeded):
        check_budget(5.01, 5.0)


def test_estimates_are_positive_and_batch_halves_claude():
    for j, m in run.JUDGE_MODELS.items():
        c = BUILDERS[j](Call(j, m, "P0", "judge", "i", "C0", "passage " * 300, "answer " * 50))
        e = estimate_call(c, batch=False)
        assert e.usd > 0 and e.input > 0
        if j in ("haiku", "sonnet"):
            assert estimate_call(c, batch=True).usd == pytest.approx(e.usd / 2)


def test_plan_counts():
    items = [{"response_id": f"r{i}", "source_passage": "p", "response": "a", "hallucinated": i % 2 == 0,
              "task_type": "QA"} for i in range(4)]
    calls, _ = run.plan(items, list(run.JUDGES), list(run.ARMS), ["C0", "C1"], {}, c4_proxy=False)
    # 4 items x 2 conditions x 4 judges x 3 requests
    assert len(calls) == 4 * 2 * 4 * 3
    kinds = {(c.judge, c.arm, c.kind) for c in calls}
    assert ("haiku", "guard", "guard") in kinds and ("haiku", "P0G", "judge") not in kinds
    assert ("jev", "P0G", "judge") in kinds


def test_c4_skipped_live_without_rewrites_and_proxied_in_dry_run():
    items = [{"response_id": "r1", "source_passage": "p", "response": "a", "hallucinated": True, "task_type": "QA"}]
    live, notes = run.plan(items, ["jev"], ["P0"], ["C4"], {}, c4_proxy=False)
    assert live == [] and "skipped" in notes[0]
    dry, notes = run.plan(items, ["jev"], ["P0"], ["C4"], {}, c4_proxy=True)
    assert len(dry) == 1 and "proxy" in notes[0]


@needs_sample
def test_live_run_over_cap_aborts_before_any_call(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(run, "execute", lambda *a, **k: called.append(1))
    monkeypatch.setattr(run, "CACHE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run.py", "--live", "--max-usd", "0.001", "--condition", "C0", "C1"])
    with pytest.raises(BudgetExceeded):
        run.main()
    assert called == []


@needs_sample
def test_pilot_items_cover_task_types_and_classes():
    sample = run.load_sample()
    p = run.pilot_items(sample)
    assert len(p) == 5 and {i["task_type"] for i in p if i["hallucinated"]} == {"Summary", "QA", "Data2txt"}
    assert any(not i["hallucinated"] for i in p)
    assert [i["response_id"] for i in p] == [i["response_id"] for i in run.pilot_items(sample)]
