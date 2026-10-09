"""fill judgement holes with the chosen llm judge on a seeded query sample, then score ndcg with noise correction.

for each language that passed the kappa gate: sample queries that have a relevant passage (seed 13),
collect the unjudged top-k passages of every run, judge them (cached), write
qrels/miracl.{lang}.filled.txt and score lower, optimistic, filled (hard labels) and corrected
(expected ndcg) per query and system. unparseable output counts as label 0 in the filled qrels.

the validation split is class balanced, so precision and npv would assume 50 percent prevalence.
the correction uses tpr and tnr instead, and estimates the share of relevant holes with rogan-gladen,
pi = clip((q + tnr - 1) / (tpr + tnr - 1), 0, 1), where q is the share of holes the judge labels 1.
by bayes a hole labelled 1 is relevant with probability pi tpr / (pi tpr + (1 - pi)(1 - tnr)), a hole
labelled 0 with pi (1 - tpr) / (pi (1 - tpr) + (1 - pi) tnr), an unparseable one with pi. if
tpr + tnr - 1 is below 0.05 the judge is too weak to correct with and corrected falls back to filled.

labels are written to results/llm_fill_labels/{lang}.csv every 50 pairs and on exit, so a rerun
only judges what is missing.
"""
import argparse
import importlib.util
import math
import os

import numpy as np
import pandas as pd

from src.utils.bootstrap import bootstrap_ci
from src.utils.config import ROOT, cfg, path, tier
from src.utils.io import read_qrels, read_run, write_qrels
from src.utils.llm import Judge
from src.utils.metrics import expected_ndcg, ndcg_at_k, optimistic_ndcg
from src.utils.text import passage_text, query_text

SAMPLE_QUERIES = 50
CHECKPOINT_EVERY = 50
MIN_YOUDEN = 0.05
TREATMENTS = ("lower", "optimistic", "filled", "corrected")
PROMPT = (ROOT / "src" / "prompts" / "judge.txt").read_text(encoding="utf-8")

_spec = importlib.util.spec_from_file_location("bounds", ROOT / "src" / "05_bounds.py")
bounds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bounds)


def sample_queries(qrels: dict, n: int, seed: int) -> list[str]:
    """uniform sample of up to n queries that have a positive judgement, sorted.
    the sample of n is inside the sample of n + k, so a bigger run reuses the cached labels."""
    qids = sorted(q for q, docs in qrels.items() if any(r > 0 for r in docs.values()))
    order = np.random.default_rng(seed).permutation(qids)
    return sorted(order[:n].tolist())


def load_runs(lang: str, runs_dir) -> dict[str, dict[str, list[str]]]:
    """runs of the configured systems that exist for this language, in config order."""
    runs = {}
    for system in cfg()["systems"]:
        file = runs_dir / f"miracl.{lang}.{system}.txt"
        if file.exists():
            runs[system] = read_run(file)
    return runs


def find_holes(runs: dict, qrels: dict, sample: list[str], k: int) -> list[tuple[str, str]]:
    """unique unjudged (qid, docid) pairs in the top k of any run, for the sampled queries."""
    return sorted({(q, d) for run in runs.values() for q in sample
                   for d in run.get(q, [])[:k] if d not in qrels[q]})


def load_labels(file, model: str, prompt_hash: str) -> dict[tuple[str, str], int | None]:
    """labels saved by an earlier run of the same model and prompt, None for unparseable."""
    if not file.exists():
        return {}
    df = pd.read_csv(file, dtype=str, keep_default_na=False)
    df = df[(df.model == model) & (df.prompt_hash == prompt_hash)]
    return {(r.qid, r.docid): (int(r.label) if r.label != "" else None) for r in df.itertuples()}


def save_labels(labels: dict, file, model: str, prompt_hash: str) -> None:
    """write through a temp file so an interrupt never leaves a half written csv."""
    rows = [{"qid": q, "docid": d, "label": "" if lab is None else lab, "model": model, "prompt_hash": prompt_hash}
            for (q, d), lab in sorted(labels.items())]
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(file.name + ".tmp")
    pd.DataFrame(rows, columns=["qid", "docid", "label", "model", "prompt_hash"]).to_csv(tmp, index=False)
    os.replace(tmp, file)


def judge_holes(judge, holes: list, labels: dict, query_of, passage_of, checkpoint,
                every: int = CHECKPOINT_EVERY) -> int:
    """judge the holes that have no label yet and add them to labels, returns how many were judged.

    checkpoint(labels) runs every `every` judgements and once more on the way out, also on an interrupt."""
    todo = [h for h in holes if h not in labels]
    if not todo:
        return 0
    try:
        for i, (q, d) in enumerate(todo, start=1):
            labels[(q, d)] = judge.judge(q, d, query_of(q), passage_of(d)).label
            if i % every == 0:
                checkpoint(labels)
    finally:
        checkpoint(labels)
    return len(todo)


def fill_qrels(qrels: dict, holes: list, labels: dict) -> dict:
    """copy of qrels with the llm label added for every hole, unparseable as 0."""
    filled = {q: dict(docs) for q, docs in qrels.items()}
    for q, d in holes:
        filled[q][d] = 0 if labels[(q, d)] is None else labels[(q, d)]
    return filled


def unparseable_rate(holes: list, labels: dict) -> float:
    if not holes:
        return 0.0
    return sum(1 for h in holes if labels[h] is None) / len(holes)


def rates_from_counts(counts) -> tuple[float, float]:
    """(tpr, tnr) from tp, fn, tn, fp, these do not depend on how many relevant passages the sample has."""
    if counts["tp"] + counts["fn"] == 0 or counts["tn"] + counts["fp"] == 0:
        raise ValueError(f"need both classes in the test split, got {dict(counts)}")
    return (counts["tp"] / (counts["tp"] + counts["fn"]), counts["tn"] / (counts["tn"] + counts["fp"]))


def positive_share(holes: list, labels: dict) -> float:
    """share of the holes the llm labelled 1, unparseable holes are left out, nan if none is left."""
    parsed = [labels[h] for h in holes if labels[h] is not None]
    return sum(parsed) / len(parsed) if parsed else float("nan")


def _posterior(hit: float, miss: float, prior: float) -> float:
    """hit / (hit + miss), the prior when both are zero."""
    total = hit + miss
    return hit / total if total > 0 else prior


def noise_model(tpr: float, tnr: float, q: float) -> dict:
    """prevalence estimate and the probability of relevance for each llm label, see the module docstring."""
    youden = tpr + tnr - 1
    if math.isnan(q) or youden < MIN_YOUDEN:
        return {"pi_hat": float("nan"), "p1": 1.0, "p0": 0.0, "p_unparseable": 0.0, "unreliable": True}
    pi = min(1.0, max(0.0, (q + tnr - 1) / youden))
    return {"pi_hat": pi, "unreliable": False, "p_unparseable": pi,
            "p1": _posterior(pi * tpr, (1 - pi) * (1 - tnr), pi),
            "p0": _posterior(pi * (1 - tpr), (1 - pi) * tnr, pi)}


def hole_probability(label: int | None, model: dict) -> float:
    if label is None:
        return model["p_unparseable"]
    return model["p1"] if label == 1 else model["p0"]


def query_rows(lang: str, runs: dict, qrels: dict, filled: dict, labels: dict, sample: list[str],
               model: dict, k: int) -> list[dict]:
    """one row per system and sampled query with the four treatments and the hole counts.

    the probabilities of a query cover the holes of every system, so the ideal ranking in the
    corrected ndcg is the same for all systems, like the ideal of the filled qrels."""
    prob = {}
    for q in sample:
        prob[q] = {d: float(v) for d, v in qrels[q].items()}
        for run in runs.values():
            for d in run.get(q, [])[:k]:
                if d not in qrels[q]:
                    prob[q][d] = hole_probability(labels[(q, d)], model)
    rows = []
    for system, run in runs.items():
        for q in sample:
            ranking, rel = run.get(q, []), qrels[q]
            top_holes = [d for d in ranking[:k] if d not in rel]
            rows.append({"lang": lang, "tier": tier(lang), "system": system, "qid": q, "has_positive": True,
                         "lower": ndcg_at_k(ranking, rel, k),
                         "optimistic": optimistic_ndcg(ranking, rel, k),
                         "filled": ndcg_at_k(ranking, filled[q], k),
                         "corrected": expected_ndcg(ranking, prob[q], k),
                         "n_holes_top10": len(top_holes),
                         "n_holes_llm_rel": sum(1 for d in top_holes if labels[(q, d)] == 1)})
    return rows


def summarise(ql: pd.DataFrame, lang_stats: dict) -> pd.DataFrame:
    """per language and system: means with bootstrap cis, holes per query, llm relevant rate."""
    n, seed = cfg()["bootstrap_resamples"], cfg()["seed"]
    rows = []
    for (lang, system), g in ql.groupby(["lang", "system"], sort=False):
        row = {"lang": lang, "tier": tier(lang), "system": system, "n_queries": len(g),
               "unique_holes": lang_stats[lang]["holes"],
               "holes_per_query": round(g.n_holes_top10.mean(), 3),
               "llm_rel_rate": round(g.n_holes_llm_rel.sum() / max(1, g.n_holes_top10.sum()), 4),
               "unparseable_rate": round(lang_stats[lang]["unparseable_rate"], 4),
               "tpr": round(lang_stats[lang]["tpr"], 4), "tnr": round(lang_stats[lang]["tnr"], 4),
               "q": round(lang_stats[lang]["q"], 4), "pi_hat": round(lang_stats[lang]["pi_hat"], 4),
               "correction_unreliable": lang_stats[lang]["correction_unreliable"]}
        for t in TREATMENTS:
            m, lo, hi = bootstrap_ci(g[t].values, n, seed)
            row[f"{t}_mean"], row[f"{t}_lo"], row[f"{t}_hi"] = round(m, 4), round(lo, 4), round(hi, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def gate_langs(val: pd.DataFrame, requested: list[str] | None) -> tuple[list[str], list[str]]:
    """(languages to fill, languages skipped) from the validation table and the requested list."""
    passed = set(val[val.passes_gate].lang)
    names = requested or list(val.lang)
    return [lang for lang in names if lang in passed], [lang for lang in names if lang not in passed]


def estimate_hours(n_pairs: int, sec_per_judgement: float) -> float:
    return n_pairs * sec_per_judgement / 3600


def prepare(lang: str, n_sample: int, k: int):
    """dev qrels, runs, sampled queries and holes for one language."""
    qrels = read_qrels(path("qrels") / f"miracl.{lang}.dev.txt")
    runs = load_runs(lang, path("runs"))
    sample = sample_queries(qrels, n_sample, cfg()["seed"])
    return qrels, runs, sample, find_holes(runs, qrels, sample, k)


def labels_file(lang: str):
    return path("results") / "llm_fill_labels" / f"{lang}.csv"


def run_language(lang: str, judge, tpr: float, tnr: float, n_sample: int, k: int, query_of, passage_of):
    """fill one language, returns its query level rows and the per language stats."""
    qrels, runs, sample, holes = prepare(lang, n_sample, k)
    if not runs:
        print(f"[fill] {lang}: no run files, skipped")
        return [], {}
    file = labels_file(lang)
    labels = load_labels(file, judge.model, judge.prompt_hash)
    todo = sum(1 for h in holes if h not in labels)
    print(f"[fill] {lang}: {len(sample)} queries, {len(holes)} holes, {todo} to judge, model={judge.model}")

    def checkpoint(current):
        save_labels(current, file, judge.model, judge.prompt_hash)
        print(f"[fill] {lang}: {sum(1 for h in holes if h in current)}/{len(holes)} labelled")

    judge_holes(judge, holes, labels, query_of, passage_of, checkpoint)
    filled = fill_qrels(qrels, holes, labels)
    write_qrels(filled, path("qrels") / f"miracl.{lang}.filled.txt")
    q = positive_share(holes, labels)
    model = noise_model(tpr, tnr, q)
    rows = query_rows(lang, runs, qrels, filled, labels, sample, model, k)
    stats = {"holes": len(holes), "unparseable_rate": unparseable_rate(holes, labels), "tpr": tpr, "tnr": tnr,
             "q": q, "pi_hat": model["pi_hat"], "correction_unreliable": model["unreliable"]}
    print(f"[fill] {lang}: unparseable rate {stats['unparseable_rate']:.4f}, tpr {tpr:.3f}, tnr {tnr:.3f}, "
          f"llm-positive share {q:.4f}")
    if model["unreliable"]:
        print(f"[fill] {lang}: correction unreliable (tpr + tnr - 1 < {MIN_YOUDEN} or no parseable hole), "
              f"corrected = filled labels")
    else:
        print(f"[fill] {lang}: estimated prevalence of relevant holes {model['pi_hat']:.4f}")
    return rows, stats


def dry_run(val: pd.DataFrame, langs: list[str], judge, n_sample: int, k: int) -> None:
    """count the holes still to judge and extrapolate the time, no judge calls."""
    fallback = val.sec_per_judgement.mean()
    total_pairs, total_hours = 0, 0.0
    for lang in langs:
        _, _, sample, holes = prepare(lang, n_sample, k)
        labels = load_labels(labels_file(lang), judge.model, judge.prompt_hash)
        todo = sum(1 for h in holes if h not in labels)
        sec = val[val.lang == lang].sec_per_judgement.iloc[0]
        sec = fallback if pd.isna(sec) else sec
        hours = estimate_hours(todo, sec)
        total_pairs, total_hours = total_pairs + todo, total_hours + hours
        print(f"[fill] {lang}: {len(sample)} queries, {len(holes)} holes, {todo} to judge, "
              f"{sec:.2f} s each, ~{hours:.2f} h")
    print(f"[fill] total: {total_pairs} pairs to judge, ~{total_hours:.2f} h")


def write_outputs(ql: pd.DataFrame, lang_stats: dict) -> None:
    results = path("results")
    ql.to_csv(results / "filled_query_level.csv", index=False)
    summ = summarise(ql, lang_stats)
    summ.to_csv(results / "filled.csv", index=False)
    bounds.gaps(ql, treatments=TREATMENTS).to_csv(results / "gaps_filled.csv", index=False)
    bounds.pairs(ql, treatments=("lower", "corrected")).to_csv(results / "pairs_filled.csv", index=False)
    print(summ.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--langs", nargs="+", help="default: every language that passed the gate")
    ap.add_argument("--sample-queries", type=int, default=SAMPLE_QUERIES)
    ap.add_argument("--dry-run", action="store_true", help="count holes and estimate the time, judge nothing")
    args = ap.parse_args(argv)
    val = pd.read_csv(path("results") / "llm_validation.csv")
    models = val.model.unique()
    if len(models) != 1:
        raise SystemExit(f"llm_validation.csv should hold one model, found {list(models)}")
    langs, skipped = gate_langs(val, args.langs)
    if skipped:
        print(f"[fill] skipped, failed the gate or not validated: {', '.join(skipped)}")
    k = cfg()["k"]
    judge = Judge(models[0], PROMPT, path("results") / "llm_labels.jsonl")
    if args.dry_run:
        dry_run(val, langs, judge, args.sample_queries, k)
        return
    frames, lang_stats = [], {}
    for lang in langs:
        row = val[val.lang == lang].iloc[0]
        rows, lang_stats[lang] = run_language(
            lang, judge, *rates_from_counts(row), args.sample_queries, k,
            lambda q, lang=lang: query_text(lang, q), lambda d, lang=lang: passage_text(lang, d))
        if rows:
            frames.append(pd.DataFrame(rows))
    if frames:
        write_outputs(pd.concat(frames, ignore_index=True), lang_stats)


if __name__ == "__main__":
    main()
