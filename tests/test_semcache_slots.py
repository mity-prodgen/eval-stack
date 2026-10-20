"""Slot guard rules on hand-written examples. None of these sentences is in semcache/pairs/pairs.jsonl."""

import pytest

from semcache.slots import extract, guard


@pytest.mark.parametrize("a,b", [
    ("Fare from Dubai to London Heathrow", "Fare from DXB to LHR"),
    ("Biz class lounge access", "Business class lounge access"),
    ("pandas wheels for 3.12", "pandas wheels for Python 3.12"),
    ("Annual fee on the Amex card", "Annual fee on the American Express card"),
    ("Wire $2k abroad", "Wire 2,000 dollars abroad"),
    ("Why can't I log in", "Why cannot I log in"),
    ("Status of my flight on 4 Nov", "Status of my flight on November 4th"),
    ("Pickup at 7pm", "Pickup at 19:00"),
    ("How do I reset my password", "What are the steps to reset my password"),
])
def test_aliases_match(a, b):
    ok, bad = guard(a, b)
    assert ok, (bad, extract(a), extract(b))


@pytest.mark.parametrize("a,b,slot", [
    ("Fare from DXB to LHR", "Fare from LHR to DXB", "origin"),
    ("Move money from savings to checking", "Move money from checking to savings", "origin"),
    ("Premium Economy seat pitch", "Economy seat pitch", "class"),
    ("Can I pay with Visa", "Can I pay with Mastercard", "entity"),
    ("Is pickup available", "Is pickup not available", "negation"),
    ("Does it run on Node 18", "Does it run on Node 20", "version"),
    ("Fee to send $500", "Fee to send $5,000", "number"),
    ("Status of my flight on 4 Nov", "Status of my flight on 5 Nov", "date"),
    ("Pickup at 7pm", "Pickup at 7am", "time"),
    ("Baggage allowance", "Baggage allowance in first class", "class"),  # presence counts
])
def test_slot_changes_block(a, b, slot):
    ok, bad = guard(a, b)
    assert not ok and slot in bad, (bad, extract(a), extract(b))


def test_money_and_versions_do_not_bleed_into_each_other():
    assert extract("Refund of 3.50 euros")["number"] == frozenset({"3.5 EUR"})
    assert "version" not in extract("Refund of 3.50 euros")
    assert extract("Is 3.12 supported")["version"] == frozenset({"3.12"})
    assert "number" not in extract("Is 3.12 supported")


def test_may_the_verb_is_not_a_date():
    assert "date" not in extract("I may need help")
    assert extract("Flights in May")["date"] == frozenset({"05"})


def test_lowercase_code_like_words_are_not_airports():
    assert "entity" not in extract("is it a sin to skip this")
