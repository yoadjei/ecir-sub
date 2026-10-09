"""local llm relevance judge via ollama, with a jsonl cache so no judgement is repeated."""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/generate"
MAX_PASSAGE_CHARS = 2000


def _call_ollama(model: str, prompt: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 4, "seed": 13},
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]


def parse_label(text: str):
    """first 0/1 digit in the output, else None (unparseable)."""
    m = re.search(r"[01]", text)
    return int(m.group()) if m else None


@dataclass
class Judgement:
    qid: str
    docid: str
    label: int | None
    raw: str
    seconds: float
    cached: bool


class Judge:
    def __init__(self, model: str, prompt_template: str, cache_path: Path):
        self.model = model
        self.template = prompt_template
        self.prompt_hash = hashlib.sha1(prompt_template.encode("utf-8")).hexdigest()[:10]
        self.cache_path = Path(cache_path)
        self.cache: dict[tuple, dict] = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    self.cache[(rec["model"], rec["prompt_hash"], rec["qid"], rec["docid"])] = rec

    def judge(self, qid: str, docid: str, query: str, passage: str) -> Judgement:
        key = (self.model, self.prompt_hash, qid, docid)
        if key in self.cache:
            r = self.cache[key]
            return Judgement(qid, docid, r["label"], r["raw"], r["seconds"], True)
        prompt = self.template.format(query=query, passage=passage[:MAX_PASSAGE_CHARS])
        t0 = time.time()
        raw = _call_ollama(self.model, prompt)
        secs = time.time() - t0
        rec = {"model": self.model, "prompt_hash": self.prompt_hash, "qid": qid, "docid": docid,
               "label": parse_label(raw), "raw": raw, "seconds": round(secs, 3)}
        self.cache[key] = rec
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return Judgement(qid, docid, rec["label"], raw, secs, False)


def confusion(gold, pred) -> dict[str, int]:
    c = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    for g, p in zip(gold, pred):
        if g == 1 and p == 1:
            c["tp"] += 1
        elif g == 1 and p == 0:
            c["fn"] += 1
        elif g == 0 and p == 0:
            c["tn"] += 1
        else:
            c["fp"] += 1
    return c


def kappa(gold, pred) -> float:
    """cohen's kappa for binary labels."""
    c = confusion(gold, pred)
    n = sum(c.values())
    if n == 0:
        return float("nan")
    po = (c["tp"] + c["tn"]) / n
    p_yes = ((c["tp"] + c["fn"]) / n) * ((c["tp"] + c["fp"]) / n)
    p_no = ((c["tn"] + c["fp"]) / n) * ((c["tn"] + c["fn"]) / n)
    pe = p_yes + p_no
    return 0.0 if pe == 1 else (po - pe) / (1 - pe)


def rates(c: dict[str, int]) -> dict[str, float]:
    """precision (ppv), recall, accuracy, npv from a confusion dict; nan where undefined."""
    def div(a, b):
        return a / b if b else float("nan")
    return {"precision": div(c["tp"], c["tp"] + c["fp"]), "recall": div(c["tp"], c["tp"] + c["fn"]),
            "accuracy": div(c["tp"] + c["tn"], sum(c.values())), "npv": div(c["tn"], c["tn"] + c["fn"])}
