"""validate the llm judge per language on judged pairs, choose the model on calibration only.

stages, run in order, each can be repeated:
  split      results/llm_splits/{lang}.csv, balanced judged pairs, a third for calibration and the rest for test
  calibrate  results/llm_calibration.csv and llm_model_choice.txt
  test       results/llm_validation.csv, agreement of the chosen model per language
a language with fewer than 75 judged positives or negatives uses the smaller count for both classes.
unparseable output counts as non-relevant and its rate is reported.
"""
import argparse
import statistics

import numpy as np
import pandas as pd

from src.utils.config import ROOT, cfg, path
from src.utils.io import read_qrels
from src.utils.llm import Judge, confusion, kappa, rates
from src.utils.text import passage_text, query_text

LANGS = ["sw", "yo", "th", "fi"]
CANDIDATES = ["qwen2.5:7b-instruct", "gemma3:4b"]
PER_CLASS = 75
GATE_KAPPA = 0.2
PROMPT = (ROOT / "src" / "prompts" / "judge.txt").read_text(encoding="utf-8")
CACHE = path("results") / "llm_labels.jsonl"


def make_split(qrels: dict, lang: str) -> pd.DataFrame:
    """balanced judged pairs with a calibration third and a test remainder, per class."""
    pos = sorted((q, d) for q, docs in qrels.items() for d, r in docs.items() if r > 0)
    neg = sorted((q, d) for q, docs in qrels.items() for d, r in docs.items() if r == 0)
    n = min(PER_CLASS, len(pos), len(neg))
    if n < 3:
        raise ValueError(f"{lang}: {len(pos)} judged positives and {len(neg)} judged negatives, need at least 3 of each")
    n_cal = n // 3
    seed = cfg()["seed"]
    rng = np.random.default_rng(seed)
    rows = []
    for rel, pairs in ((1, pos), (0, neg)):
        for i, idx in enumerate(rng.choice(len(pairs), n, replace=False)):
            q, d = pairs[idx]
            rows.append({"lang": lang, "qid": q, "docid": d, "rel": rel,
                         "split": "calibration" if i < n_cal else "test"})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_note(lang: str, df: pd.DataFrame) -> str:
    """empty for a full split, else a line saying how short it is."""
    n = int((df.rel == 1).sum())
    if n >= PER_CLASS:
        return ""
    cal = int(((df.rel == 1) & (df.split == "calibration")).sum())
    return f"{lang}: only {n} judged pairs per class available, wanted {PER_CLASS}; {cal} per class for calibration, {n - cal} for test"


def load_split(lang: str) -> pd.DataFrame:
    """split from the cached csv, built from the dev qrels on first use."""
    out = path("results") / "llm_splits" / f"{lang}.csv"
    if out.exists():
        return pd.read_csv(out, dtype=str).assign(rel=lambda d: d.rel.astype(int))
    df = make_split(read_qrels(path("qrels") / f"miracl.{lang}.dev.txt"), lang)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def score(judge, lang: str, df: pd.DataFrame, query_of, passage_of) -> dict:
    """agreement of the judge with the qrels labels; query_of(qid) and passage_of(docid) give the text."""
    gold, pred, secs, unparseable = [], [], [], 0
    for r in df.itertuples():
        j = judge.judge(str(r.qid), str(r.docid), query_of(str(r.qid)), passage_of(str(r.docid)))
        if j.label is None:
            unparseable += 1
            pred.append(0)
        else:
            pred.append(j.label)
        gold.append(int(r.rel))
        if not j.cached:
            secs.append(j.seconds)
    c = confusion(gold, pred)
    out = {"lang": lang, "model": judge.model, "n": len(df), "kappa": round(kappa(gold, pred), 4),
           "unparseable_rate": round(unparseable / len(df), 4),
           "sec_per_judgement": round(statistics.mean(secs), 2) if secs else float("nan")}
    out.update({k: round(v, 4) for k, v in rates(c).items()})
    out.update(c)
    return out


def choose_model(cal: pd.DataFrame, candidates: list[str]) -> str:
    """highest mean calibration kappa, ties go to the earlier candidate."""
    means = cal.groupby("model").kappa.mean()
    return max(candidates, key=lambda m: means[m])


def score_lang(judge: Judge, lang: str, split: str) -> dict:
    df = load_split(lang)
    return score(judge, lang, df[df.split == split],
                 lambda q: query_text(lang, q), lambda d: passage_text(lang, d))


def calibrate(langs: list[str]) -> None:
    rows = []
    for model in CANDIDATES:
        judge = Judge(model, PROMPT, CACHE)
        for lang in langs:
            rows.append(score_lang(judge, lang, "calibration"))
            print(rows[-1])
    cal = pd.DataFrame(rows)
    cal.to_csv(path("results") / "llm_calibration.csv", index=False)
    choice = choose_model(cal, CANDIDATES)
    means = cal.groupby("model").kappa.mean().sort_values(ascending=False)
    (path("results") / "llm_model_choice.txt").write_text(
        f"chosen_model: {choice}\nmean_calibration_kappa:\n{means.to_string()}\nprompt_hash: {judge.prompt_hash}\n",
        encoding="utf-8")
    print(f"[llm] frozen model: {choice}")


def validate_test(langs: list[str]) -> None:
    lines = (path("results") / "llm_model_choice.txt").read_text(encoding="utf-8").splitlines()
    judge = Judge(lines[0].split(": ")[1], PROMPT, CACHE)
    rows = []
    for lang in langs:
        rows.append(score_lang(judge, lang, "test"))
        rows[-1]["passes_gate"] = rows[-1]["kappa"] >= GATE_KAPPA
        print(rows[-1])
    pd.DataFrame(rows).to_csv(path("results") / "llm_validation.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "calibrate", "test"])
    ap.add_argument("--langs", nargs="*", default=LANGS)
    args = ap.parse_args()
    if args.stage == "split":
        for lang in args.langs:
            df = load_split(lang)
            print(lang, df.groupby(["split", "rel"]).size().to_dict())
            note = split_note(lang, df)
            if note:
                print(f"[llm] short split, {note}")
    elif args.stage == "calibrate":
        calibrate(args.langs)
    else:
        validate_test(args.langs)


if __name__ == "__main__":
    main()
