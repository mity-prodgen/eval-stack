"""Request shapes and response parsing for every judge, through mocked HTTP (no network)."""

import json

import httpx2 as httpx
import pytest

from common import HAIKU55_MODEL, HAIKU_MODEL, JEV_MODEL, LUNA_MODEL, SONNET_MODEL
from jev_harness.cache import DiskCache
from judges import (Call, build_claude, build_jev, build_luna, parse_claude, parse_jev, parse_luna,
                    run_claude_batch, run_claude_sync, run_luna)

PASSAGE = {"name": "Cafe", "hours": "9-5"}


def call(judge, model, arm="P0", kind="judge", cond="C0", response="It opens at 9."):
    return Call(judge, model, arm, kind, "item1", cond, PASSAGE, response)


# ---------------------------------------------------------------- builders
def test_jev_request_shape_and_cache_key_depends_on_arm():
    a, b = build_jev(call("jev", JEV_MODEL, "P0")), build_jev(call("jev", JEV_MODEL, "P1"))
    assert a.payload["state"] == {"source_passage": PASSAGE, "response": "It opens at 9."}
    assert a.key != b.key and len(a.key) == 32


def test_haiku_request_is_post2_shape_with_temperature_zero():
    c = build_claude(call("haiku", HAIKU_MODEL))
    p = c.payload
    assert p["temperature"] == 0 and "thinking" not in p and p["max_tokens"] == 200
    assert p["messages"][0]["content"].startswith("SOURCE PASSAGE:\n{")


def test_sonnet_request_has_no_temperature_and_caches_passage_block():
    p = build_claude(call("sonnet", SONNET_MODEL)).payload
    assert "temperature" not in p
    assert p["thinking"] == {"type": "between_tools"}
    blocks = p["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"} and "RESPONSE:" not in blocks[0]["text"]
    assert blocks[1]["text"] == "RESPONSE:\nIt opens at 9."
    # same characters as the post-2 single-string message
    from common import judge_user_text
    assert blocks[0]["text"] + blocks[1]["text"] == judge_user_text(PASSAGE, "It opens at 9.")


def test_haiku55_request_has_no_temperature_thinking_disabled_and_cached_passage():
    p = build_claude(call("haiku55", HAIKU55_MODEL)).payload
    assert p["model"] == "claude-haiku-5-5" and "temperature" not in p
    assert p["thinking"] == {"type": "disabled"}
    assert p["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_claude_guard_uses_guard_system_prompt():
    from common import CLAUDE_GUARD_SYSTEM
    assert build_claude(call("haiku", HAIKU_MODEL, "guard", "guard")).payload["system"] == CLAUDE_GUARD_SYSTEM


# ---------------------------------------------------------------- parsers
def test_parse_jev_ok_and_guard():
    raw = {"model": JEV_MODEL, "answers": {"supported": {"noul": 0.3}, "guard": {"noul": 0.9}}, "usage": {"input_tokens": 1000}}
    r = parse_jev(raw, "P0G")
    assert r["status"] == "ok" and r["p_pass"] == 0.3 and r["p_guard"] == 0.9
    assert r["cost_usd"] == pytest.approx(1000 * 0.042 / 1e6)


def test_parse_jev_missing_answer_is_parse_error_not_dropped():
    assert parse_jev({"answers": {}}, "P0")["status"] == "parse_error"


def test_parse_luna_refusal_and_partial_refusal():
    raw = {"model": LUNA_MODEL, "usage": {"input_tokens": 100, "output_tokens": 0},
           "answers": [{"type": "refusal", "name": "supported"}]}
    assert parse_luna(raw, "P0")["status"] == "refusal"
    raw["answers"] = [{"type": "predicate", "name": "supported", "probability": 0.7}, {"type": "refusal", "name": "guard"}]
    r = parse_luna(raw, "P0G")
    assert r["status"] == "refusal" and r["p_pass"] == 0.7 and "guard=refusal" in r["error"]


def test_parse_claude_judgment_refusal_and_garbage():
    base = {"model": HAIKU_MODEL, "usage": {"input_tokens": 500, "output_tokens": 16}, "stop_reason": "end_turn"}
    ok = parse_claude({**base, "content": [{"type": "text", "text": '{"hallucinated": true, "confidence": 80}'}]}, "judge", batch=False, requested_model=HAIKU_MODEL)
    assert ok["status"] == "ok" and ok["p_pass"] == pytest.approx(0.1)
    assert ok["cost_usd"] == pytest.approx((500 * 1 + 16 * 5) / 1e6)
    batch = parse_claude({**base, "content": [{"type": "text", "text": '{"hallucinated": false, "confidence": 100}'}]}, "judge", batch=True, requested_model=HAIKU_MODEL)
    assert batch["p_pass"] == 1.0 and batch["cost_usd"] == pytest.approx(ok["cost_usd"] / 2)
    ref = parse_claude({**base, "stop_reason": "refusal", "content": []}, "judge", batch=False, requested_model=HAIKU_MODEL)
    assert ref["status"] == "refusal"
    bad = parse_claude({**base, "content": [{"type": "text", "text": "The response is fine."}]}, "judge", batch=False, requested_model=HAIKU_MODEL)
    assert bad["status"] == "parse_error" and bad["ok"] is True  # cached and counted, not retried forever
    g = parse_claude({**base, "content": [{"type": "text", "text": '{"guard": true, "confidence": 90}'}]}, "guard", batch=False, requested_model=HAIKU_MODEL)
    assert g["p_guard"] == pytest.approx(0.95) and g["p_pass"] is None


def test_cost_uses_requested_model_not_response_model_string():
    raw = {"model": "claude-haiku-5-5", "stop_reason": "end_turn",
           "content": [{"type": "text", "text": '{"hallucinated": false, "confidence": 50}'}],
           "usage": {"input_tokens": 1000, "output_tokens": 16}}
    r = parse_claude(raw, "judge", batch=False, requested_model=HAIKU55_MODEL)
    assert r["cost_usd"] == pytest.approx((1000 * 0.10 + 16 * 0.50) / 1e6)


def test_haiku55_long_prompt_tier():
    from jev_harness.prices import usage_cost
    assert usage_cost(HAIKU55_MODEL, 100_000) == pytest.approx(0.01)
    assert usage_cost(HAIKU55_MODEL, 100_001) == pytest.approx(100_001 * 0.50 / 1e6)


def test_sonnet_cost_counts_cache_write_and_read():
    raw = {"model": SONNET_MODEL, "stop_reason": "end_turn",
           "content": [{"type": "text", "text": '{"hallucinated": false, "confidence": 50}'}],
           "usage": {"input_tokens": 100, "output_tokens": 16, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 0}}
    r = parse_claude(raw, "judge", batch=True, requested_model=SONNET_MODEL)
    assert r["cost_usd"] == pytest.approx((100 * 2 + 16 * 10 + 1000 * 2.5) * 0.5 / 1e6)


# ---------------------------------------------------------------- mocked SDK round trips
def test_luna_round_trip_through_openai_sdk(tmp_path):
    from openai import OpenAI
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json={
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": "supported", "probability": 0.2},
                        {"type": "predicate", "name": "guard", "probability": 0.8}],
            "usage": {"input_tokens": 300, "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                      "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 300}})

    client = OpenAI(api_key="test", base_url="http://mock/v1", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    c = build_luna(call("luna", LUNA_MODEL, "P0G"))
    out = run_luna([c], DiskCache(tmp_path, "t"), client=client)[c.key]
    assert seen[0][0] == "/v1/decisions"
    assert seen[0][1]["model"] == "gpt-6-luna" and [q["name"] for q in seen[0][1]["questions"]] == ["supported", "guard"]
    assert out["status"] == "ok" and out["p_pass"] == 0.2 and out["p_guard"] == 0.8 and out["cached"] is False
    # second run hits the cache, no request
    out2 = run_luna([c], DiskCache(tmp_path, "t"), client=client)[c.key]
    assert out2["cached"] is True and len(seen) == 1


def _msg(model, text):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 400, "output_tokens": 16}}


def test_haiku_sync_sends_temperature_zero_on_the_wire(tmp_path):
    import anthropic
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_msg(HAIKU_MODEL, '{"hallucinated": false, "confidence": 70}'))

    client = anthropic.Anthropic(api_key="test", base_url="http://mock", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    c = build_claude(call("haiku", HAIKU_MODEL))
    out = run_claude_sync([c], DiskCache(tmp_path, "t"), client=client)[c.key]
    assert seen[0]["temperature"] == 0 and seen[0]["model"] == HAIKU_MODEL
    assert out["status"] == "ok" and out["p_pass"] == pytest.approx(0.85)


def test_claude_batch_round_trip_and_resume_file(tmp_path):
    import anthropic
    seen = {"create": None}
    batch = {"id": "msgbatch_1", "type": "message_batch", "processing_status": "ended",
             "request_counts": {"processing": 0, "succeeded": 2, "errored": 0, "canceled": 0, "expired": 0},
             "created_at": "2026-10-13T00:00:00Z", "expires_at": "2026-10-14T00:00:00Z",
             "ended_at": "2026-10-13T00:05:00Z", "archived_at": None, "cancel_initiated_at": None,
             "results_url": "http://mock/v1/messages/batches/msgbatch_1/results"}
    calls = [build_claude(call("sonnet", SONNET_MODEL)), build_claude(call("haiku", HAIKU_MODEL, cond="C1"))]

    def handler(req):
        if req.method == "POST" and req.url.path == "/v1/messages/batches":
            seen["create"] = json.loads(req.content)
            return httpx.Response(200, json=batch)
        if req.url.path.endswith("/results"):
            lines = [json.dumps({"custom_id": calls[0].key, "result": {"type": "succeeded", "message": _msg(SONNET_MODEL, '{"hallucinated": true, "confidence": 100}')}}),
                     json.dumps({"custom_id": calls[1].key, "result": {"type": "errored", "error": {"type": "error", "error": {"type": "api_error", "message": "boom"}}}})]
            return httpx.Response(200, text="\n".join(lines), headers={"content-type": "application/binary"})
        return httpx.Response(200, json=batch)

    client = anthropic.Anthropic(api_key="test", base_url="http://mock", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    cache = DiskCache(tmp_path, "t")
    out = run_claude_batch(calls, cache, tmp_path / "batches", poll_s=0, client=client, log=lambda *_: None)
    reqs = {r["custom_id"]: r["params"] for r in seen["create"]["requests"]}
    assert reqs[calls[1].key]["temperature"] == 0 and "temperature" not in reqs[calls[0].key]
    assert out[calls[0].key]["status"] == "ok" and out[calls[0].key]["batch"] is True
    assert out[calls[1].key]["status"] == "api_error"
    assert cache.get(calls[0].key) is not None and cache.get(calls[1].key) is None  # errors are retried next run
    meta = json.loads((tmp_path / "batches" / "msgbatch_1.json").read_text())
    assert meta["collected"] is True
    assert meta["requests"][calls[0].key] == {"model": SONNET_MODEL, "kind": "judge"}
