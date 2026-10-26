"""freshness (post 3): frozen data, rendering, prompts, request shapes, parsers, estimates, ledger. No network."""

import hashlib
import json

import pytest

from freshness import build_items, calls, estimate, run
from freshness.common import (DECIDERS, DOCSETS, ITEMS, JOB1_QUESTION, JOB2_QUESTION, MANIFEST, NOT_REACHABLE,
                              RAG_PROMPT, WIKI_IDS, WIKI_RAW, gen_prompt)
from harness.cache import DiskCache
from harness.ledger import Ledger

needs_wiki = pytest.mark.skipif(not WIKI_RAW.exists(), reason="run scripts/download_wikicontradict.sh")
FIRST = json.loads(ITEMS.open().readline())


# ---------------------------------------------------------------- data
def test_frozen_items_match_manifest_and_rebuild():
    m = json.loads(MANIFEST.read_text())
    for path, name in ((DOCSETS, "docsets.json"), (ITEMS, "items.jsonl"), (WIKI_IDS, "wiki_ids.json")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m[name], name
    rebuilt = build_items.build_authored(build_items.load_docsets())
    assert "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in rebuilt) == ITEMS.read_text()


def test_docsets_and_items_have_no_value_leaks():
    assert build_items.check_docsets(build_items.load_docsets()) == []
    for it in (json.loads(l) for l in ITEMS.open()):
        assert build_items.check_item(it) == [], it["id"]


def test_item_counts_and_balance():
    items = [json.loads(l) for l in ITEMS.open()]
    kinds = [i["kind"] for i in items]
    assert kinds.count("conflict") == 101 and kinds.count("control") == 40
    fmts = [i["date_format"] for i in items if i["kind"] == "conflict"]
    assert max(fmts.count(f) for f in set(fmts)) - min(fmts.count(f) for f in set(fmts)) <= 1
    by_id = {i["id"]: i for i in items}
    for k in (i for i in items if i["kind"] == "control"):
        twin = by_id[k["twin"]]
        assert k["question"] == twin["question"] and k["current"] == k["stale"] == twin["current"]
        assert k["chunks"]["new"] == twin["chunks"]["new"] and k["chunks"]["old"] != twin["chunks"]["old"]


def test_date_formats_render_where_promised():
    for it in (json.loads(l) for l in ITEMS.open()):
        new = it["chunks"]["new"]
        if it["date_format"] == "version":
            assert it["new_version"] in new.splitlines()[0] and not any(y in new for y in ("2024", "2025", "2026"))
        elif it["date_format"] == "body":
            assert "As of " in new and it["new_version"] not in new
        else:
            assert it["new_version"] not in new and new.splitlines()[1][:4] in ("Effe", "Last", "Publ")


def test_has_matches_whole_tokens():
    assert build_items.has("8", "8 or more") and not build_items.has("8", "28 days")
    assert not build_items.has("16", "Xcode 16.2") and build_items.has("16.2", "Xcode 16.2 or later")
    assert build_items.has("$75", "over $75.") and not build_items.has("$75", "$750")


@needs_wiki
def test_wiki_sample_is_fixed_and_distinct():
    rows = build_items.load_wiki_rows()
    spec = build_items.sample_wiki(rows)
    assert spec == json.loads(WIKI_IDS.read_text())["items"]
    wi = build_items.wiki_items(spec, rows)
    assert len(wi) == 20 and all(w["chunks"]["new"] != w["chunks"]["old"] for w in wi)


# ---------------------------------------------------------------- prompts and questions
def test_rag_prompt_is_langchain_hub_verbatim():
    assert RAG_PROMPT.startswith("You are an assistant for question-answering tasks.")
    assert RAG_PROMPT.endswith("\nQuestion: {question} \nContext: {context} \nAnswer:")
    assert gen_prompt("Q?", "CTX").endswith("Question: Q? \nContext: CTX \nAnswer:")


def test_chunk_orders():
    a, b = calls.chunks_for(FIRST, "new_first"), calls.chunks_for(FIRST, "old_first")
    assert a[0] == b[1] == FIRST["chunks"]["new"] and a[1] == b[0] == FIRST["chunks"]["old"] and a[2:] == b[2:]
    assert calls.context_for(FIRST, "old_first").startswith(FIRST["chunks"]["old"] + "\n\n" + FIRST["chunks"]["new"])


def test_every_decider_gets_the_same_wording_and_state():
    states, texts = set(), {"job1": set(), "job2": set()}
    for d in DECIDERS:
        for job in ("job1", "job2"):
            p = calls.build_decision(FIRST, "new_first", d.name, job, "ANSWER").payload
            if d.api == "luna":
                (q,) = p["questions"]
                assert q["type"] == "predicate" and q["name"] == job
                texts[job].add(q["instructions"])
                states.add((job, p["input"]))
            else:
                assert set(p["questions"]) == {job} and p["questions"][job]["type"] == "noul"
                assert set(p["questions"][job]) == {"type", "instructions"}       # no criteria: same wording as Luna
                texts[job].add(p["questions"][job]["instructions"])
                states.add((job, p["state"]))
                assert ("model" in p) == d.send_model
    assert texts == {"job1": {JOB1_QUESTION}, "job2": {JOB2_QUESTION}}
    assert len(states) == 2   # one job-1 state and one job-2 state shared by all seven models


def test_job2_state_contains_the_answer_and_job1_does_not():
    s1 = calls.build_decision(FIRST, "new_first", "jev", "job1").payload["state"]
    s2 = calls.build_decision(FIRST, "new_first", "jev", "job2", "MY ANSWER").payload["state"]
    assert "ANSWER:" not in s1 and s2.endswith("ANSWER:\nMY ANSWER")


def test_microsoft_recorded_as_not_reachable():
    assert "Microsoft-Decision-1" in NOT_REACHABLE and all("microsoft" not in d.name for d in DECIDERS)
    assert {d.model for d in DECIDERS} >= {"pplx-decider-v1.2", "pplx-decider-flash-v1.2", "jev-1.14.0"}


def test_repeat_tags_make_fresh_keys():
    k0 = calls.build_decision(FIRST, "new_first", "clef", "job1").key
    assert calls.build_decision(FIRST, "new_first", "clef", "job1", repeat="noise1").key != k0
    assert calls.build_decision(FIRST, "new_first", "strands", "job1").key != calls.build_decision(
        FIRST, "new_first", "clef", "job1").key


# ---------------------------------------------------------------- request shapes
def test_generator_and_baseline_shapes():
    h = calls.build_gen(FIRST, "new_first", "haiku55").payload
    assert h["thinking"] == {"type": "disabled"} and "temperature" not in h and len(h["messages"]) == 1
    g = calls.build_gen(FIRST, "new_first", "gemma").payload
    assert g["temperature"] == 1.0 and g["top_p"] == 0.95 and g["extra_body"] == {"top_k": 64}
    assert g["messages"][0]["role"] == "user" and len(g["messages"]) == 1     # no system prompt -> no <|think|>
    b = calls.build_baseline(FIRST, "new_first", "A").payload
    assert b["model"] == "claude-sonnet-5-5" and b["messages"][0]["content"][0]["cache_control"]


# ---------------------------------------------------------------- parsers
def test_parse_label_text():
    ok = '{"label": "stale", "conflict_noted": false, "timeline": true}'
    assert calls.parse_label_text(ok, wiki=False) == ("stale", False, True)
    assert calls.parse_label_text('x {"label": "B", "conflict_noted": true}', wiki=True) == ("b", True, None)
    for bad in ('{"label": "maybe", "conflict_noted": false, "timeline": false}',
                '{"label": "current", "conflict_noted": "no", "timeline": false}',
                '{"label": "current", "conflict_noted": false}',                     # timeline missing
                '{"label": "current", "conflict_noted": false, "timeline": "yes"}', "none"):
        with pytest.raises((ValueError, KeyError)):
            calls.parse_label_text(bad, wiki=False)


def test_timeline_field_only_in_the_stale_current_prompt():
    from freshness.common import LABEL_SYSTEM, WIKI_LABEL_SYSTEM
    assert '"timeline": true|false' in LABEL_SYSTEM and "timeline" not in WIKI_LABEL_SYSTEM
    # the hand-checked label definitions are unchanged: no extra rule about dated answers in the blends line
    assert "even if the newer value comes last" not in LABEL_SYSTEM


def test_timeline_status():
    assert run.timeline_status({"timeline": True}, {"timeline": True}) == "timeline"
    assert run.timeline_status({"timeline": False}, {"timeline": False}) == "plain"
    assert run.timeline_status({"timeline": True}, {"timeline": False}) == "split"
    assert run.timeline_status({"timeline": None}, {"timeline": True}) == "n/a"


def test_parse_gen_strips_gemma_empty_thought_and_flags_empty():
    d = calls.parse_gen({"choices": [{"message": {"content": "<|channel>thought\n<channel|>21 days."},
                                      "finish_reason": "stop"}], "usage": {"prompt_tokens": 1000, "completion_tokens": 10}},
                        {"provider": "deepinfra", "model": "google/gemma-4-31B-it"})
    assert d["text"] == "21 days." and d["cost_usd"] == pytest.approx((1000 * 0.20 + 10 * 0.40) / 1e6)
    e = calls.parse_gen({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}], "usage": {}},
                        {"provider": "deepinfra", "model": "google/gemma-4-31B-it"})
    assert e["status"] == "empty"


def test_parse_decision_all_shapes():
    s1 = {"answers": {"job1": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 1000}}
    assert calls.parse_decision(s1, {"step": "job1", "model": "jev-1.14.0"})["p_yes"] == 0.8
    cf = {"result": s1, "success": True}
    r = calls.parse_decision(cf, {"step": "job1", "model": "clef"})
    assert r["p_yes"] == 0.8 and r["cost_usd"] == pytest.approx(1000 * 0.24 / 1e6)
    lu = {"answers": [{"name": "job2", "type": "predicate", "probability": 0.1}], "usage": {"input_tokens": 10}}
    assert calls.parse_decision(lu, {"step": "job2", "model": "gpt-6-luna"})["p_yes"] == 0.1
    st = {"answers": {"job1": {"type": "noul", "noul": 0.3}}, "usage": {"input_tokens": 500}, "latency_ms": 489.0}
    r = calls.parse_decision(st, {"step": "job1", "model": "x", "local": True})
    assert r["cost_usd"] == 0.0 and r["server_latency_ms"] == 489.0
    assert calls.parse_decision({"answers": {}}, {"step": "job1", "model": "jev-1.14.0"})["status"] == "parse_error"


def test_parse_baseline():
    c = calls.parse_baseline({"content": [{"type": "text", "text": '{"hallucinated": false, "confidence": 80}'}],
                              "usage": {"input_tokens": 100, "output_tokens": 10}}, {"model": "claude-sonnet-5-5"})
    assert c["p_pass"] == pytest.approx(0.9)


def test_string_match_and_kappa():
    it = {"kind": "conflict", "current": "21 days", "stale": "30 days"}
    assert run.string_match(it, "Within 21 days.") == "current"
    assert run.string_match(it, "21 days (was 30 days)") == "both"
    assert run.kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


# ---------------------------------------------------------------- cache, estimate, ledger
def test_api_errors_are_not_cached(tmp_path):
    cache = DiskCache(tmp_path, "t")
    cache.get_or_call("k", lambda: {"ok": False, "status": "api_error"})
    assert cache.get("k") is None


def test_ledger_caps_the_whole_post(tmp_path):
    led = Ledger(tmp_path / "spend.json", 5.0)
    assert led.allowance(5.0) == 5.0
    led.record("pilot", "gen", 1.25, 1.5)
    led.record("full", "label", 3.0)
    assert led.spent() == pytest.approx(4.25) and led.allowance(5.0) == pytest.approx(0.75)
    with pytest.raises(SystemExit):
        estimate.check_budget(1.0, led.allowance(5.0))


@needs_wiki
def test_dry_run_plan_size_and_cost():
    items = build_items.load_items()
    by_id = {i["id"]: i for i in items}
    assert len(items) == 161
    gens = run.plan_gens(items, ["haiku55", "gemma"])
    assert len({c.key for c in gens}) == 161 * 2 * 2
    answers = [{"item_id": c.item_id, "order": c.order, "gen": c.gen, "text": f"A {c.key}"} for c in gens]
    names = [d.name for d in DECIDERS]
    j1, j2 = run.plan_job1(items, names), run.plan_job2(answers, by_id, names)
    assert len({c.key for c in j1}) == 161 * 2 * 7 and len({c.key for c in j2}) == 141 * 2 * 2 * 7
    assert all(by_id[c.item_id]["kind"] != "wiki" for c in j2)
    lat = run.plan_latency(items, names, 100)
    assert sum("warmup" not in c.repeat for c in lat) == 700
    total = sum(e.usd for e in estimate.estimate(gens + j1 + j2).values())
    assert 0 < total < 5


def test_timeline_sheet_takes_timeline_and_split_answers_and_keeps_human_labels(tmp_path, monkeypatch):
    import csv
    monkeypatch.setattr(run, "TIMELINE_SHEET", tmp_path / "timeline_labels.csv")
    items = [json.loads(l) for l in ITEMS.open()]
    by_id = {i["id"]: i for i in items}
    conflict = [i for i in items if i["kind"] == "conflict"][:3]
    control = next(i for i in items if i["kind"] == "control")
    answers = [{"item_id": i["id"], "order": "new_first", "gen": "gemma", "text": f"A {i['id']}"}
               for i in conflict + [control]]
    flags = {conflict[0]["id"]: (True, True), conflict[1]["id"]: (True, False), conflict[2]["id"]: (False, False),
             control["id"]: (True, True)}
    labels = {}
    for iid, (a, b) in flags.items():
        for who, t in (("sonnet", a), ("sol", b)):
            labels[(iid, "new_first", "gemma", who)] = {"status": "ok", "label": "blends", "timeline": t}
    run.write_timeline_sheet(answers, labels, by_id)
    rows = list(csv.DictReader(open(run.TIMELINE_SHEET)))
    assert [r["item_id"] for r in rows] == [conflict[0]["id"], conflict[1]["id"]]       # timeline + split, no control
    assert [r["labelers_timeline"] for r in rows] == ["timeline", "split"]
    rows[0]["your_label"] = "stale"
    with open(run.TIMELINE_SHEET, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    run.write_timeline_sheet(answers, labels, by_id)
    assert list(csv.DictReader(open(run.TIMELINE_SHEET)))[0]["your_label"] == "stale"
