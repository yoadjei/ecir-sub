"""ciral: the same runs scored under shallow and deep judgements for the same queries.

test set a has two qrels over the same queries, test-a (about 18 judgements per query) and
test-a-pools (about 90 per query). test set b has only the shallow kind and is used to check
the runs against the published two-click nDCG@20.

systems:
  bm25qt            bm25 on the native index with human translated queries
  bm25dt            bm25 on the machine translated english index with english queries
  rrf_bm25          reciprocal rank fusion of bm25qt and bm25dt
  mdpr, afriberta   dense runs, only where the index fits in memory
  rrf_dt_afriberta  fusion of bm25dt and afriberta, where both exist

usage: PYTHONPATH=. python src/08_ciral.py [--langs yo sw] [--search]
--search runs the pyserini cli for run files that are missing, without it only existing runs are used.
"""
import argparse
import subprocess
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from src.utils.bootstrap import bootstrap_ci, holm, paired_randomisation_p
from src.utils.config import cfg, path
from src.utils.io import pyserini_qrels, read_run, read_run_scores, write_qrels, write_run
from src.utils.metrics import judged_at_k, ndcg_at_k

LANGS = ["ha", "so", "sw", "yo"]
BASE_SYSTEMS = ["bm25qt", "bm25dt", "mdpr", "afriberta"]
FUSIONS = {"rrf_bm25": ["bm25qt", "bm25dt"], "rrf_dt_afriberta": ["bm25dt", "afriberta"]}
SYSTEMS = ["bm25qt", "bm25dt", "rrf_bm25", "mdpr", "afriberta", "rrf_dt_afriberta"]
DEPTHS = ["shallow", "deep"]
METRICS = ["ndcg10", "ndcg20"]
TOLERANCE = 0.01
HITS = 1000

# published two-click reproduction scores, nDCG@20 on test set b
REFERENCE = {
    ("bm25qt", "ha"): 0.2121, ("bm25qt", "so"): 0.1725, ("bm25qt", "sw"): 0.1727, ("bm25qt", "yo"): 0.3459,
    ("bm25dt", "ha"): 0.2124, ("bm25dt", "so"): 0.2186, ("bm25dt", "sw"): 0.2582, ("bm25dt", "yo"): 0.3700,
    ("mdpr", "ha"): 0.0397, ("mdpr", "so"): 0.0635, ("mdpr", "sw"): 0.1227, ("mdpr", "yo"): 0.1458,
    ("afriberta", "ha"): 0.2028, ("afriberta", "so"): 0.1682, ("afriberta", "sw"): 0.2166, ("afriberta", "yo"): 0.1157,
}


def run_path(lang: str, system: str, testset: str = "a"):
    suffix = "" if testset == "a" else ".testb"
    return path("runs") / f"ciral.{lang}.{system}{suffix}.txt"


def cli_command(lang: str, system: str, testset: str) -> list[str]:
    topics = f"ciral-v1.0-{lang}-test-{testset}"
    out = str(run_path(lang, system, testset))
    if system == "bm25qt":
        return [sys.executable, "-m", "pyserini.search.lucene", "--language", lang, "--topics", f"{topics}-native",
                "--index", f"ciral-v1.0-{lang}", "--output", out, "--bm25", "--hits", str(HITS),
                "--batch-size", "32", "--threads", "4"]
    if system == "bm25dt":
        return [sys.executable, "-m", "pyserini.search.lucene", "--topics", topics,
                "--index", f"ciral-v1.0-{lang}-en", "--output", out, "--bm25", "--hits", str(HITS),
                "--batch-size", "32", "--threads", "4"]
    dense = {"mdpr": ("castorini/mdpr-tied-pft-msmarco", "mdpr-tied-pft-msmarco"),
             "afriberta": ("castorini/afriberta-dpr-pft-msmarco-ft-latin-mrtydi",
                           "afriberta-dpr-ptf-msmarco-ft-latin-mrtydi")}
    if system in dense:
        encoder, suffix = dense[system]
        return [sys.executable, "-m", "pyserini.search.faiss", "--encoder-class", "auto", "--encoder", encoder,
                "--topics", topics, "--index", f"ciral-v1.0-{lang}-{suffix}", "--output", out,
                "--hits", str(HITS), "--batch-size", "32", "--threads", "4"]
    raise ValueError(f"system {system} is not produced by the cli")


def rrf_fuse(runs: list[dict[str, list[str]]], rrf_k: int = 60, depth: int = HITS) -> dict[str, list[str]]:
    """reciprocal rank fusion of rankings, ties broken by docid so the output is stable."""
    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for run in runs:
        for qid, ranking in run.items():
            for rank, docid in enumerate(ranking[:depth], start=1):
                scores[qid][docid] += 1.0 / (rrf_k + rank)
    return {qid: [d for d, _ in sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))] for qid, s in scores.items()}


def query_level(lang: str, system: str, run: dict[str, list[str]], shallow: dict, deep: dict) -> pd.DataFrame:
    """one row per query judged under both qrels, a query without results scores zero."""
    rows = []
    for qid in sorted(set(shallow) & set(deep)):
        ranking = run.get(qid, [])
        rows.append({"lang": lang, "system": system, "qid": qid,
                     "judged10_shallow": judged_at_k(ranking, shallow[qid], 10),
                     "judged10_deep": judged_at_k(ranking, deep[qid], 10),
                     "ndcg10_shallow": ndcg_at_k(ranking, shallow[qid], 10),
                     "ndcg10_deep": ndcg_at_k(ranking, deep[qid], 10),
                     "ndcg20_shallow": ndcg_at_k(ranking, shallow[qid], 20),
                     "ndcg20_deep": ndcg_at_k(ranking, deep[qid], 20)})
    return pd.DataFrame(rows)


QL_VALUE_COLUMNS = ["judged10_shallow", "judged10_deep", "ndcg10_shallow", "ndcg10_deep",
                    "ndcg20_shallow", "ndcg20_deep"]


def summarise(ql: pd.DataFrame) -> pd.DataFrame:
    n, seed = cfg()["bootstrap_resamples"], cfg()["seed"]
    rows = []
    for (lang, system), g in ql.groupby(["lang", "system"], sort=False):
        row = {"lang": lang, "system": system, "n_queries": len(g)}
        for col in QL_VALUE_COLUMNS:
            m, lo, hi = bootstrap_ci(g[col].values, n, seed)
            row[f"{col}_mean"], row[f"{col}_lo"], row[f"{col}_hi"] = round(m, 4), round(lo, 4), round(hi, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_systems(summary: pd.DataFrame, column: str) -> pd.Series:
    """rank of each system by a mean column of one language, 1 is best, ties share the lower rank."""
    return summary.set_index("system")[column].rank(ascending=False, method="min").astype(int)


def add_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """rank columns (1 is best) per language for the four ndcg means."""
    out = summary.copy()
    for column in ["ndcg10_shallow_mean", "ndcg10_deep_mean", "ndcg20_shallow_mean", "ndcg20_deep_mean"]:
        rank_column = column.replace("_mean", "_rank")
        out[rank_column] = 0
        for lang in out.lang.unique():
            mask = out.lang == lang
            out.loc[mask, rank_column] = out.loc[mask, "system"].map(rank_systems(out[mask], column))
    return out


def pair_table(ql: pd.DataFrame, systems: list[str], metric: str) -> pd.DataFrame:
    """pairwise paired tests per language and depth, holm over the pairs of a language and depth.

    sign_flip: the mean difference has opposite signs under shallow and deep.
    flip_one_sig / flip_both_sig: a sign flip where at least one / both depths are significant.
    sig_change: significant (holm p < 0.05) under one depth and not under the other."""
    n_perm, seed = cfg()["permutations"], cfg()["seed"]
    rows = []
    for lang in list(dict.fromkeys(ql.lang)):
        sub = ql[ql.lang == lang]
        present = [s for s in systems if s in set(sub.system)]
        per_depth = {}
        for depth in DEPTHS:
            wide = sub.pivot(index="qid", columns="system", values=f"{metric}_{depth}")
            block = []
            for a, b in combinations(present, 2):
                x, y = wide[a].values, wide[b].values
                block.append({"sys_a": a, "sys_b": b, "mean_diff": float(np.mean(x - y)),
                              "p_raw": paired_randomisation_p(x, y, n_perm, seed)})
            for r, p in zip(block, holm([r["p_raw"] for r in block])):
                r["p_holm"] = p
                r["sig"] = bool(p < 0.05)
            per_depth[depth] = block
        for sh, dp in zip(per_depth["shallow"], per_depth["deep"]):
            flip = bool(sh["mean_diff"] * dp["mean_diff"] < 0)
            rows.append({"lang": lang, "metric": metric, "sys_a": sh["sys_a"], "sys_b": sh["sys_b"],
                         "mean_diff_shallow": round(sh["mean_diff"], 4), "p_raw_shallow": round(sh["p_raw"], 4),
                         "p_holm_shallow": round(sh["p_holm"], 4), "sig_shallow": sh["sig"],
                         "mean_diff_deep": round(dp["mean_diff"], 4), "p_raw_deep": round(dp["p_raw"], 4),
                         "p_holm_deep": round(dp["p_holm"], 4), "sig_deep": dp["sig"],
                         "sign_flip": flip, "flip_one_sig": bool(flip and (sh["sig"] or dp["sig"])),
                         "flip_both_sig": bool(flip and sh["sig"] and dp["sig"]),
                         "sig_change": bool(sh["sig"] != dp["sig"])})
    return pd.DataFrame(rows)


def qrels_stats(shallow: dict, deep: dict) -> dict:
    """sizes of the two qrels over the same queries, their overlap and label agreement on it."""
    def pairs(q):
        return {(qid, d): int(r) for qid, docs in q.items() for d, r in docs.items()}
    sp, dp = pairs(shallow), pairs(deep)
    shared = sp.keys() & dp.keys()
    return {"queries": len(set(shallow) & set(deep)),
            "shallow_judgements": len(sp), "deep_judgements": len(dp),
            "shallow_positives": sum(1 for r in sp.values() if r > 0),
            "deep_positives": sum(1 for r in dp.values() if r > 0),
            "shallow_in_deep": len(shared),
            "label_agree": sum(1 for p in shared if (sp[p] > 0) == (dp[p] > 0)),
            "queries_no_positive_shallow": sum(1 for docs in shallow.values() if not any(r > 0 for r in docs.values())),
            "queries_no_positive_deep": sum(1 for docs in deep.values() if not any(r > 0 for r in docs.values()))}


def pool_share(run: dict[str, list[str]], qrels: dict, k: int) -> float:
    """share of the judged pairs that sit in the top k of the run, over all judged pairs."""
    judged = sum(len(docs) for docs in qrels.values())
    hit = 0
    for qid, docs in qrels.items():
        top = set(run.get(qid, [])[:k])
        hit += sum(1 for d in docs if d in top)
    return hit / judged if judged else 0.0


def qrels_counts(qrels: dict) -> dict:
    return {"queries": len(qrels), "judgements": sum(len(d) for d in qrels.values()),
            "positives": sum(1 for d in qrels.values() for r in d.values() if r > 0)}


def trec_ndcg20(qrels: dict, run_file) -> float:
    """nDCG@20 over all queries in the qrels, the number the published reproduction reports (trec_eval -c)."""
    import ir_measures
    from ir_measures import nDCG
    agg = ir_measures.calc_aggregate([nDCG @ 20], qrels, read_run_scores(run_file))
    return float(agg[nDCG @ 20])


def ensure_runs(lang: str, testsets: str, search: bool) -> None:
    for system in BASE_SYSTEMS:
        for testset in testsets:
            out = run_path(lang, system, testset)
            if out.exists() or not search:
                continue
            print("[ciral] " + " ".join(cli_command(lang, system, testset)))
            subprocess.run(cli_command(lang, system, testset), check=True)


def write_fusions(lang: str) -> None:
    """fusion runs on test set a, built from the base runs that exist."""
    for name, parts in FUSIONS.items():
        files = [run_path(lang, p) for p in parts]
        if not all(f.exists() for f in files):
            continue
        fused = rrf_fuse([read_run(f) for f in files])
        write_run({q: [(d, 1.0 / (r + 1)) for r, d in enumerate(docs)] for q, docs in fused.items()},
                  run_path(lang, name), name)


def reproduction(lang: str, qrels_b: dict) -> list[dict]:
    rows = []
    for system in SYSTEMS:
        out = run_path(lang, system, "b")
        if not out.exists() or (system, lang) not in REFERENCE:
            continue
        ours = trec_ndcg20(qrels_b, out)
        ref = REFERENCE[(system, lang)]
        rows.append({"lang": lang, "system": system, "n_queries": len(qrels_b), "ndcg20": round(ours, 4),
                     "reference": ref, "diff": round(ours - ref, 4), "within_tolerance": abs(ours - ref) <= TOLERANCE})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--langs", nargs="+", choices=LANGS, default=LANGS)
    ap.add_argument("--search", action="store_true", help="run the pyserini cli for missing run files")
    args = ap.parse_args(argv)
    results = path("results")
    ql_frames, repro_rows, stat_rows, pool_rows = [], [], [], []
    for lang in args.langs:
        ensure_runs(lang, "ab", args.search)
        write_fusions(lang)
        qrels_b = pyserini_qrels(f"ciral-v1.0-{lang}-test-b")
        repro_rows += reproduction(lang, qrels_b)
        shallow = pyserini_qrels(f"ciral-v1.0-{lang}-test-a")
        deep = pyserini_qrels(f"ciral-v1.0-{lang}-test-a-pools")
        write_qrels(shallow, path("qrels") / f"ciral.{lang}.shallow.txt")
        write_qrels(deep, path("qrels") / f"ciral.{lang}.deep.txt")
        stat_rows.append({"lang": lang, **qrels_stats(shallow, deep),
                          **{f"testb_{k}": v for k, v in qrels_counts(qrels_b).items()}})
        for system in SYSTEMS:
            if not run_path(lang, system).exists():
                continue
            run = read_run(run_path(lang, system))
            ql_frames.append(query_level(lang, system, run, shallow, deep))
            pool_rows.append({"lang": lang, "system": system, "k": 20,
                              "shallow_in_top20": round(pool_share(run, shallow, 20), 4),
                              "deep_in_top20": round(pool_share(run, deep, 20), 4)})
    ql = pd.concat(ql_frames, ignore_index=True)
    ql.to_csv(results / "ciral_query_level.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(results / "ciral_qrels.csv", index=False)
    pd.DataFrame(repro_rows).to_csv(results / "ciral_repro.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(results / "ciral_pool_share.csv", index=False)
    summ = add_ranks(summarise(ql))
    summ.to_csv(results / "ciral.csv", index=False)
    pairs = pd.concat([pair_table(ql, SYSTEMS, m) for m in METRICS], ignore_index=True)
    pairs.to_csv(results / "ciral_pairs.csv", index=False)
    print(pd.DataFrame(repro_rows).to_string(index=False))
    print(summ[["lang", "system", "n_queries", "judged10_shallow_mean", "judged10_deep_mean", "ndcg10_shallow_mean",
                "ndcg10_deep_mean", "ndcg20_shallow_mean", "ndcg20_deep_mean"]].to_string(index=False))
    print(pairs[["lang", "metric", "sys_a", "sys_b", "mean_diff_shallow", "p_holm_shallow", "mean_diff_deep",
                 "p_holm_deep", "sign_flip", "flip_one_sig", "sig_change"]].to_string(index=False))


if __name__ == "__main__":
    main()
