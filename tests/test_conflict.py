"""conflict (post 3): frozen data, rendering, prompts, request shapes, parsers, estimates, budget gate. No network."""

import hashlib
import json

import pytest

from conflict import build_items, calls, estimate, run
from conflict.common import (DOCSETS, F1_RULE, ITEMS, J1_SENTENCE, MANIFEST, PRIMARY_LABELER, RAG_PROMPT, WIKI_IDS,
                             WIKI_RAW, f2_notice, gen_prompt)
from harness.cache import DiskCache

needs_wiki = pytest.mark.skipif(not WIKI_RAW.exists(), reason="run scripts/download_wikicontradict.sh")


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
    items = [json.loads(l) for l in ITEMS.open()]
    for it in items:
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


# ---------------------------------------------------------------- prompts
def test_rag_prompt_is_langchain_hub_verbatim_and_arms_only_add_one_sentence():
    assert RAG_PROMPT.startswith("You are an assistant for question-answering tasks.")
    assert RAG_PROMPT.endswith("\nQuestion: {question} \nContext: {context} \nAnswer:")
    f0, f1 = gen_prompt("Q?", "CTX", "F0"), gen_prompt("Q?", "CTX", "F1")
    assert f1.replace(" " + F1_RULE, "", 1) == f0
    n = f2_notice("2")
    assert gen_prompt("Q?", "CTX", "F2jev", n).replace(" " + n, "", 1) == f0
    assert gen_prompt("Q?", "CTX", "F2luna", None) == f0      # no notice -> the F0 prompt (cache hit on F0)
    assert f2_notice("none") == "Note: the retrieved sources give different answers to this question."


def test_chunk_orders():
    it = json.loads(ITEMS.open().readline())
    a, b = calls.chunks_for(it, "new_first"), calls.chunks_for(it, "old_first")
    assert a[0] == b[1] == it["chunks"]["new"] and a[1] == b[0] == it["chunks"]["old"] and a[2:] == b[2:]
    assert calls.context_for(it, "old_first").startswith("[1] " + it["chunks"]["old"])


def test_j1_adds_the_rubric_sentence_to_every_judge():
    it = json.loads(ITEMS.open().readline())
    for j in calls.JUDGES:
        j0 = json.dumps(calls.build_judge(it, "new_first", "A", j, "J0").payload)
        j1 = json.dumps(calls.build_judge(it, "new_first", "A", j, "J1").payload)
        assert J1_SENTENCE not in j0 and J1_SENTENCE in j1


def test_primary_labeler_is_never_the_generator():
    for g, lb in PRIMARY_LABELER.items():
        assert calls.GENERATORS[g] != calls.LABELERS[lb]


# ---------------------------------------------------------------- request shapes
def test_generator_request_shapes():
    it = json.loads(ITEMS.open().readline())
    h = calls.build_gen(it, "new_first", "haiku55", "F0").payload
    assert h["thinking"] == {"type": "disabled"} and "temperature" not in h and len(h["messages"]) == 1
    s = calls.build_gen(it, "new_first", "sonnet", "F0").payload
    assert s["thinking"] == {"type": "between_tools"}
    o = calls.build_gen(it, "new_first", "sol", "F0").payload
    assert o["reasoning"] == {"effort": "low"} and isinstance(o["input"], str)
    q = calls.build_gen(it, "new_first", "qwen", "F0").payload
    assert q["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False} and q["temperature"] == 0.7
    assert calls.build_gen(it, "new_first", "sol", "F0", repeat=1).key != calls.build_gen(it, "new_first", "sol", "F0").key


def test_precheck_shapes():
    it = json.loads(ITEMS.open().readline())
    j = calls.build_precheck(it, "old_first", "jev").payload
    assert set(j["questions"]) == {"conflict", "newest"} and j["questions"]["newest"]["type"] == "choice"
    lu = calls.build_precheck(it, "old_first", "luna").payload
    assert [q["type"] for q in lu["questions"]] == ["predicate", "choice"]
    assert lu["questions"][0]["instructions"].startswith("Do these chunks give different answers to the question?")


# ---------------------------------------------------------------- parsers
def test_parse_label_text():
    assert calls.parse_label_text('{"label": "stale", "conflict_noted": false}', wiki=False) == ("stale", False)
    assert calls.parse_label_text('x {"label": "B", "conflict_noted": true}', wiki=True) == ("b", True)
    for bad in ('{"label": "maybe", "conflict_noted": false}', '{"label": "current", "conflict_noted": "no"}', "none"):
        with pytest.raises((ValueError, KeyError)):
            calls.parse_label_text(bad, wiki=False)


def test_parse_gen_all_providers():
    a = calls.parse_gen({"content": [{"type": "text", "text": " 21 days. "}], "stop_reason": "end_turn",
                         "usage": {"input_tokens": 1000, "output_tokens": 10}},
                        {"provider": "anthropic", "model": "claude-sonnet-5-5", "batch": True})
    assert a["status"] == "ok" and a["text"] == "21 days." and a["cost_usd"] == pytest.approx((2000 + 100) * 0.5 / 1e6)
    o = calls.parse_gen({"status": "completed", "output": [{"type": "reasoning"}, {"type": "message", "content": [
        {"type": "output_text", "text": "21 days"}]}], "usage": {"input_tokens": 1000, "output_tokens": 50,
        "output_tokens_details": {"reasoning_tokens": 40}, "input_tokens_details": {"cached_tokens": 0}}},
        {"provider": "openai", "model": "gpt-6.1-sol"})
    assert o["text"] == "21 days" and o["usage"]["reasoning"] == 40
    d = calls.parse_gen({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 0}},
                        {"provider": "deepinfra", "model": "Qwen/Qwen3.8-27B"})
    assert d["status"] == "empty"
    r = calls.parse_gen({"content": [], "stop_reason": "refusal", "usage": {}},
                        {"provider": "anthropic", "model": "claude-haiku-5-5"})
    assert r["status"] == "refusal"


def test_parse_precheck_and_judge():
    j = calls.parse_precheck({"answers": {"conflict": {"noul": 0.8}, "newest": {"choice": "2", "probabilities": {}}},
                              "usage": {"input_tokens": 500}}, {"provider": "jev", "model": "jev-1.13.0"})
    assert j["p_conflict"] == 0.8 and j["newest"] == "2"
    lu = calls.parse_precheck({"answers": [{"name": "conflict", "type": "predicate", "probability": 0.1},
                                           {"name": "newest", "type": "choice", "choice": "none"}],
                               "usage": {"input_tokens": 500}}, {"provider": "luna", "model": "gpt-6-luna"})
    assert lu["p_conflict"] == 0.1 and lu["newest"] == "none"
    c = calls.parse_judge({"content": [{"type": "text", "text": '{"hallucinated": false, "confidence": 80}'}],
                           "usage": {"input_tokens": 100, "output_tokens": 10}},
                          {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"})
    assert c["p_pass"] == pytest.approx(0.9)


def test_notice_rule():
    assert run.notice_for(None, None) is None and run.notice_for(None, "1") == f2_notice("1")
    assert run.notice_for({"status": "ok", "p_conflict": 0.49, "newest": "2"}, None) is None
    assert run.notice_for({"status": "ok", "p_conflict": 0.5, "newest": "2"}, None) == f2_notice("2")
    assert run.notice_for({"status": "parse_error", "p_conflict": None}, None) is None


def test_string_match_and_kappa():
    it = {"kind": "conflict", "current": "21 days", "stale": "30 days"}
    assert run.string_match(it, "Within 21 days.") == "current"
    assert run.string_match(it, "21 days (was 30 days)") == "both"
    assert run.kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


# ---------------------------------------------------------------- cache, estimate, budget
def test_api_errors_are_not_cached(tmp_path):
    cache = DiskCache(tmp_path, "t")
    res, _ = cache.get_or_call("k", lambda: {"ok": False, "status": "api_error"})
    assert cache.get("k") is None


@needs_wiki
def test_dry_run_plan_size_and_budget_gate():
    items = build_items.load_items()
    assert len(items) == 161
    pre = run.plan_prechecks(items)
    assert len({c.key for c in pre}) == 161 * 2 * 2
    gens = run.plan_gens(items, list(calls.GENERATORS), ["F0", "F1", "F2jev", "F2luna"], {}, run.noise_items(items),
                         assume_fire="1")
    assert len({c.key for c in gens}) == 4 * (161 * 2 * 4 + 40 * 2)
    total = sum(e.usd for e in estimate.estimate(gens).values())
    assert total > 0
    with pytest.raises(SystemExit):
        estimate.check_budget(total, 0.01)
