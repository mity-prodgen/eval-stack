"""Post 1 noise baseline: --repeat gives fresh cache keys; noise metrics on synthetic runs."""

import run
from noise import noise_metrics


def _items():
    return [{"response_id": f"r{i}", "source_passage": "p", "response": f"answer {i}", "hallucinated": i < 2,
             "task_type": "QA"} for i in range(4)]


def test_repeat_zero_keeps_original_keys_and_repeats_differ():
    base, _ = run.plan(_items(), ["jev"], ["P0"], ["C0"], {}, c4_proxy=False)
    keys0 = [c.key for c in base]
    r0 = [c.key for c in run.with_repeat(run.plan(_items(), ["jev"], ["P0"], ["C0"], {}, c4_proxy=False)[0], 0)]
    r1 = [c.key for c in run.with_repeat(run.plan(_items(), ["jev"], ["P0"], ["C0"], {}, c4_proxy=False)[0], 1)]
    r2 = [c.key for c in run.with_repeat(run.plan(_items(), ["jev"], ["P0"], ["C0"], {}, c4_proxy=False)[0], 2)]
    assert r0 == keys0
    assert not set(r1) & set(keys0) and not set(r1) & set(r2) and len(set(r1)) == len(r1)


def test_noise_metrics():
    hall = {"r0": True, "r1": True, "r2": False, "r3": False}
    run0 = {("jev", "P0", i): v for i, v in {"r0": False, "r1": False, "r2": True, "r3": True}.items()}
    run1 = {("jev", "P0", i): v for i, v in {"r0": True, "r1": False, "r2": True, "r3": True}.items()}
    run2 = dict(run0)
    m = noise_metrics([run0, run1, run2], hall)["jev|P0"]
    # r0 changes in 2 of 3 run pairs, the rest never: mean = (2/3) / 4
    assert m["changed"]["n"] == 4 and abs(m["changed"]["ci"][0] - (2 / 3) / 4) < 1e-9
    assert m["any_change"]["ci"][0] == 0.25
    # wrong items failing in run 0: r0, r1 -> re-run verdicts r0: pass, fail; r1: fail, fail -> 1/4
    assert m["noise_flip"]["n"] == 4 and m["noise_flip"]["ci"][0] == 0.25
