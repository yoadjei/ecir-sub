import json

import pytest

from src.utils import llm


def test_parse_label():
    assert llm.parse_label("1") == 1
    assert llm.parse_label(" 0\n") == 0
    assert llm.parse_label("Answer: 1") == 1
    assert llm.parse_label("relevant") is None


def test_cohen_kappa_perfect_and_chance():
    assert llm.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert llm.kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_confusion_counts():
    c = llm.confusion(gold=[1, 1, 0, 0, 1], pred=[1, 0, 0, 1, 1])
    assert c == {"tp": 2, "fn": 1, "tn": 1, "fp": 1}


def test_rates_from_confusion():
    r = llm.rates({"tp": 2, "fn": 1, "tn": 1, "fp": 1})
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["accuracy"] == pytest.approx(3 / 5)
    assert r["npv"] == pytest.approx(1 / 2)


def test_rates_are_nan_when_undefined():
    r = llm.rates({"tp": 0, "fn": 0, "tn": 3, "fp": 0})
    assert r["precision"] != r["precision"]
    assert r["recall"] != r["recall"]


def test_cache_roundtrip(tmp_path, monkeypatch):
    cache = tmp_path / "labels.jsonl"
    calls = {"n": 0}

    def fake_call(model, prompt):
        calls["n"] += 1
        return "1"

    monkeypatch.setattr(llm, "_call_ollama", fake_call)
    j = llm.Judge(model="fake", prompt_template="Q {query} P {passage}", cache_path=cache)
    a = j.judge("q", "d", "what?", "text")
    b = j.judge("q", "d", "what?", "text")
    assert a.label == 1 and b.label == 1
    assert calls["n"] == 1  # second call served from cache
    assert len(cache.read_text(encoding="utf-8").strip().splitlines()) == 1
    rec = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
    assert rec["qid"] == "q" and rec["docid"] == "d" and rec["model"] == "fake"


def test_cache_is_reloaded_from_disk(tmp_path, monkeypatch):
    cache = tmp_path / "labels.jsonl"
    monkeypatch.setattr(llm, "_call_ollama", lambda model, prompt: "0")
    llm.Judge("fake", "Q {query} P {passage}", cache).judge("q", "d", "what?", "text")

    def fail(model, prompt):
        raise AssertionError("cache miss")

    monkeypatch.setattr(llm, "_call_ollama", fail)
    again = llm.Judge("fake", "Q {query} P {passage}", cache).judge("q", "d", "what?", "text")
    assert again.label == 0 and again.cached


def test_cache_key_includes_model_and_prompt(tmp_path, monkeypatch):
    cache = tmp_path / "labels.jsonl"
    calls = []
    monkeypatch.setattr(llm, "_call_ollama", lambda model, prompt: calls.append(model) or "1")
    llm.Judge("a", "Q {query} P {passage}", cache).judge("q", "d", "x", "y")
    llm.Judge("b", "Q {query} P {passage}", cache).judge("q", "d", "x", "y")
    llm.Judge("a", "R {query} P {passage}", cache).judge("q", "d", "x", "y")
    assert calls == ["a", "b", "a"]


def test_unparseable_output_gives_none_label(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_call_ollama", lambda model, prompt: "maybe")
    j = llm.Judge("fake", "Q {query} P {passage}", tmp_path / "labels.jsonl")
    assert j.judge("q", "d", "x", "y").label is None


def test_passage_is_truncated_in_prompt(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "_call_ollama", lambda model, prompt: seen.append(prompt) or "1")
    j = llm.Judge("fake", "{passage}", tmp_path / "labels.jsonl")
    j.judge("q", "d", "x", "a" * 5000)
    assert len(seen[0]) == 2000
