"""Generic batch runners: Anthropic Message Batches and the OpenAI Batch API (both 50% off), cached per request.

Each request is (key, params, meta). `key` is the DiskCache key and the batch custom_id; `meta` is any JSON-able
dict the parser needs (model, step, ...). It is written next to the batch id before polling, so an interrupted run
resumes the same batch on the next start instead of paying twice, and a resumed batch parses and prices exactly as
the original would have. parse(raw_response_dict, meta) returns the record to cache; records with ok=False are not
cached (same rule as DiskCache.get_or_call), so they are retried on the next run.

persuasion/judges.py keeps its own (older, judge-specific) Anthropic batch runner; this module is the general form.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .cache import DiskCache

Request = tuple[str, dict[str, Any], dict[str, Any]]
Parser = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _cached(requests: list[Request], cache: DiskCache) -> dict[str, dict]:
    out = {}
    for key, _, _ in requests:
        hit = cache.get(key)
        if hit is not None:
            out[key] = {**hit, "cached": True}
    return out


def _store(key: str, res: dict, cache: DiskCache, out: dict) -> None:
    if res.get("ok") is not False:
        res.setdefault("cached_at", time.time())
        cache.set(key, res)
    out[key] = {**res, "cached": False}


def _pending(batch_dir: Path, wanted: set[str]) -> list[tuple[Path, dict]]:
    found = []
    for f in sorted(batch_dir.glob("*.json")):
        meta = json.loads(f.read_text())
        if not meta.get("collected") and set(meta["keys"]) & wanted:
            found.append((f, meta))
    return found


def _chunks(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


# =====================================================================================================
# Anthropic Message Batches
# =====================================================================================================
def anthropic_batch(requests: list[Request], cache: DiskCache, batch_dir: Path, parse: Parser, client: Any = None,
                    poll_s: float = 30.0, max_per_batch: int = 10_000, log: Callable[[str], None] = print) -> dict[str, dict]:
    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=4)
    batch_dir.mkdir(parents=True, exist_ok=True)
    out = _cached(requests, cache)
    for f, meta in _pending(batch_dir, {k for k, _, _ in requests} - set(out)):
        log(f"resuming Anthropic batch {meta['id']}")
        _anthropic_collect(client, meta, f, cache, out, parse, poll_s, log)
    todo = list({k: (k, p, m) for k, p, m in requests if k not in out}.values())
    for part in _chunks(todo, max_per_batch):
        b = client.messages.batches.create(requests=[{"custom_id": k, "params": p} for k, p, _ in part])
        meta = {"id": b.id, "provider": "anthropic", "keys": [k for k, _, _ in part], "created": time.time(),
                "collected": False, "meta": {k: m for k, _, m in part}}
        f = batch_dir / f"{b.id}.json"
        f.write_text(json.dumps(meta))
        log(f"submitted Anthropic batch {b.id} with {len(part)} requests")
        _anthropic_collect(client, meta, f, cache, out, parse, poll_s, log)
    return out


def _anthropic_collect(client, meta, meta_file, cache, out, parse, poll_s, log) -> None:
    while True:
        b = client.messages.batches.retrieve(meta["id"])
        if b.processing_status == "ended":
            break
        log(f"  batch {meta['id']}: {b.processing_status} {b.request_counts}")
        time.sleep(poll_s)
    for entry in client.messages.batches.results(meta["id"]):
        key, r = entry.custom_id, entry.result
        m = meta["meta"][key]
        if r.type == "succeeded":
            _store(key, parse(r.message.to_dict(), {**m, "batch": True}), cache, out)
        else:
            out[key] = {"ok": False, "status": "api_error", "error": f"batch {r.type}: {getattr(r, 'error', None)}",
                        "cached": False}
    meta["collected"] = True
    meta_file.write_text(json.dumps(meta))


# =====================================================================================================
# OpenAI Batch API (/v1/responses)
# =====================================================================================================
TERMINAL = ("completed", "failed", "expired", "cancelled")


def openai_batch(requests: list[Request], cache: DiskCache, batch_dir: Path, parse: Parser, client: Any = None,
                 endpoint: str = "/v1/responses", poll_s: float = 30.0, max_per_batch: int = 5_000,
                 log: Callable[[str], None] = print) -> dict[str, dict]:
    if client is None:
        from openai import OpenAI
        client = OpenAI(max_retries=4, timeout=120.0)
    batch_dir.mkdir(parents=True, exist_ok=True)
    out = _cached(requests, cache)
    for f, meta in _pending(batch_dir, {k for k, _, _ in requests} - set(out)):
        log(f"resuming OpenAI batch {meta['id']}")
        _openai_collect(client, meta, f, cache, out, parse, poll_s, log)
    todo = list({k: (k, p, m) for k, p, m in requests if k not in out}.values())
    for part in _chunks(todo, max_per_batch):
        lines = "".join(json.dumps({"custom_id": k, "method": "POST", "url": endpoint, "body": p}) + "\n"
                        for k, p, _ in part)
        path = batch_dir / f"input-{int(time.time() * 1000)}.jsonl"
        path.write_text(lines)
        with path.open("rb") as fh:
            up = client.files.create(file=fh, purpose="batch")
        b = client.batches.create(input_file_id=up.id, endpoint=endpoint, completion_window="24h")
        meta = {"id": b.id, "provider": "openai", "keys": [k for k, _, _ in part], "created": time.time(),
                "collected": False, "input_file": path.name, "meta": {k: m for k, _, m in part}}
        f = batch_dir / f"{b.id}.json"
        f.write_text(json.dumps(meta))
        log(f"submitted OpenAI batch {b.id} with {len(part)} requests")
        _openai_collect(client, meta, f, cache, out, parse, poll_s, log)
    return out


def _openai_collect(client, meta, meta_file, cache, out, parse, poll_s, log) -> None:
    while True:
        b = client.batches.retrieve(meta["id"])
        if b.status in TERMINAL:
            break
        log(f"  batch {meta['id']}: {b.status} {getattr(b, 'request_counts', None)}")
        time.sleep(poll_s)
    seen = set()
    for file_id in (b.output_file_id, b.error_file_id):
        if not file_id:
            continue
        for line in client.files.content(file_id).text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["custom_id"]
            seen.add(key)
            resp = row.get("response") or {}
            m = meta["meta"][key]
            if resp.get("status_code") == 200 and not row.get("error"):
                _store(key, parse(resp["body"], {**m, "batch": True}), cache, out)
            else:
                out[key] = {"ok": False, "status": "api_error", "cached": False,
                            "error": f"batch {b.status}: {row.get('error') or resp.get('body')}"}
    for key in set(meta["keys"]) - seen:
        out[key] = {"ok": False, "status": "api_error", "cached": False, "error": f"batch {b.status}: no result"}
    meta["collected"] = True
    meta["status"] = b.status
    meta_file.write_text(json.dumps(meta))
