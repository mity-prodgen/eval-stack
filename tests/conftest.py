"""Test setup: import path for persuasion/, and a hard block on network access.

Any socket connection attempted during tests raises, so the suite cannot make a live API call
even by accident. SDK clients are exercised through httpx.MockTransport instead.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "persuasion"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("network access attempted during tests")
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TYPESAFE_API_KEY", "CO_API_KEY", "COHERE_API_KEY",
                "VOYAGE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPINFRA_API_KEY",
                "PERPLEXITY_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"):
        monkeypatch.delenv(var, raising=False)
