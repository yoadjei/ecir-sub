"""nDCG@10 under lower, condensed and optimistic treatments, plus gap and ranking identifiability.

treatments:
  lower             unjudged passages count as non-relevant (the standard score)
  condensed         unjudged passages are removed from the ranking
  optimistic        exact maximum over labellings of the unjudged top-10 passages, never below lower
  all_relevant      every unjudged top-10 passage relevant, ideal recomputed (not a bound)
  optimistic_fixed  like all_relevant but with the ideal from the original qrels
a gap is identified when its 95% ci excludes 0 under both lower and optimistic with the same sign.
a pair is stable when the holm-adjusted p is below 0.05 under both with the same sign of the difference.
languages or systems without a run file are skipped.
"""
import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from src.utils.bootstrap import bootstrap_ci, bootstrap_diff_ci, bootstrap_group_means_diff, holm, paired_randomisation_p
from src.utils.config import cfg, langs, path, tier
from src.utils.io import read_qrels, read_run
from src.utils.metrics import all_relevant_ndcg, condensed_ndcg, ndcg_at_k, optimistic_fixed_ndcg, optimistic_ndcg

TREATMENTS = ["lower", "condensed", "optimistic", "all_relevant", "optimistic_fixed"]


def available_systems(lang: str) -> list[str]:
    return [s for s in cfg()["systems"] if (path("runs") / f"miracl.{lang}.{s}.txt").exists()]


def selected_langs(requested: list[str] | None = None) -> list[str]:
    """requested (or all configured) languages that have at least one run file."""
    return [lang for lang in (requested or langs()) if available_systems(lang)]


def query_level(lang_codes: list[str] | None = None) -> pd.DataFrame:
    k = cfg()["k"]
    rows = []
    for lang in selected_langs(lang_codes):
        qrels = read_qrels(path("qrels") / f"miracl.{lang}.dev.txt")
        for system in available_systems(lang):
            run = read_run(path("runs") / f"miracl.{lang}.{system}.txt")
            for qid, rel in qrels.items():
                r = run.get(qid, [])
                rows.append({"lang": lang, "tier": tier(lang), "system": system, "qid": qid,
                             "has_positive": any(v > 0 for v in rel.values()),
                             "lower": ndcg_at_k(r, rel, k), "condensed": condensed_ndcg(r, rel, k),
                             "optimistic": optimistic_ndcg(r, rel, k),
                             "all_relevant": all_relevant_ndcg(r, rel, k),
                             "optimistic_fixed": optimistic_fixed_ndcg(r, rel, k)})
    return pd.DataFrame(rows)


def summarise(ql: pd.DataFrame) -> pd.DataFrame:
    n, seed = cfg()["bootstrap_resamples"], cfg()["seed"]
    rows = []
    for (lang, system), g in ql.groupby(["lang", "system"], sort=False):
        row = {"lang": lang, "tier": tier(lang), "system": system,
               "contributor": cfg()["systems"][system]["contributor"], "n_queries": len(g)}
        for t in TREATMENTS:
            m, lo, hi = bootstrap_ci(g[t].values, n, seed)
            row[f"{t}_mean"], row[f"{t}_lo"], row[f"{t}_hi"] = round(m, 4), round(lo, 4), round(hi, 4)
        row["width"] = round(row["optimistic_mean"] - row["lower_mean"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def violations(ql: pd.DataFrame) -> pd.DataFrame:
    v = ql[ql.optimistic < ql.lower - 1e-9]
    return v[["lang", "system", "qid", "lower", "optimistic"]]


def all_relevant_below_lower(ql: pd.DataFrame) -> pd.DataFrame:
    """queries where the all-relevant variant scores under the standard ndcg (informational)."""
    return ql[ql.all_relevant < ql.lower - 1e-9][["lang", "system", "qid", "lower", "all_relevant"]]


def gaps(ql: pd.DataFrame, treatments: tuple[str, ...] = ("lower", "condensed", "optimistic")) -> pd.DataFrame:
    """tier-level and pairwise language gaps per system under each treatment.

    identifiability is always judged on 'lower' vs 'optimistic', which must be in `treatments`."""
    n, seed = cfg()["bootstrap_resamples"], cfg()["seed"]
    rows = []
    for system in cfg()["systems"]:
        sub = ql[ql.system == system]
        present = [lang for lang in langs() if lang in set(sub.lang)]
        low = [lang for lang in present if tier(lang) == "low"]
        upper = [lang for lang in present if tier(lang) != "low"]
        if low and upper:
            for t in treatments:
                ga = [sub[sub.lang == lang][t].values for lang in upper]
                gb = [sub[sub.lang == lang][t].values for lang in low]
                d, lo, hi = bootstrap_group_means_diff(ga, gb, n, seed)
                rows.append({"system": system, "gap": f"tier:{'+'.join(upper)} - {'+'.join(low)}", "treatment": t,
                             "delta": round(d, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
        for hi_l, lo_l in cfg()["gap_pairs"]:
            if hi_l in present and lo_l in present:
                for t in treatments:
                    d, lo, hi = bootstrap_diff_ci(sub[sub.lang == hi_l][t].values, sub[sub.lang == lo_l][t].values, n, seed)
                    rows.append({"system": system, "gap": f"{hi_l} - {lo_l}", "treatment": t,
                                 "delta": round(d, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["excludes_zero"] = (df.ci_lo > 0) | (df.ci_hi < 0)
    df["sign"] = np.sign(df.delta)
    ident = []
    for (system, gap), g in df.groupby(["system", "gap"]):
        lw = g[g.treatment == "lower"].iloc[0]
        op = g[g.treatment == "optimistic"].iloc[0]
        ok = bool(lw["excludes_zero"] and op["excludes_zero"] and lw["sign"] == op["sign"])
        ident.append({"system": system, "gap": gap, "identified": ok})
    return df.merge(pd.DataFrame(ident), on=["system", "gap"])


def pairs(ql: pd.DataFrame, treatments: tuple[str, str] = ("lower", "optimistic")) -> pd.DataFrame:
    """pairwise system tests within each language under two treatments; stability compares the two."""
    n_perm, seed = cfg()["permutations"], cfg()["seed"]
    rows = []
    for lang in [lang for lang in langs() if lang in set(ql.lang)]:
        sub = ql[ql.lang == lang]
        systems = [s for s in cfg()["systems"] if s in set(sub.system)]
        for t in treatments:
            wide = sub.pivot(index="qid", columns="system", values=t)
            block = []
            for a, b in combinations(systems, 2):
                x, y = wide[a].values, wide[b].values
                block.append({"lang": lang, "treatment": t, "sys_a": a, "sys_b": b,
                              "mean_diff": round(float(np.mean(x - y)), 4),
                              "p_raw": paired_randomisation_p(x, y, n_perm, seed)})
            adj = holm([r["p_raw"] for r in block])
            for r, p in zip(block, adj):
                r["p_holm"] = round(p, 4)
                r["sig"] = p < 0.05
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    stab = []
    for (lang, a, b), g in df.groupby(["lang", "sys_a", "sys_b"]):
        lw = g[g.treatment == treatments[0]].iloc[0]
        op = g[g.treatment == treatments[1]].iloc[0]
        same_sign = np.sign(lw["mean_diff"]) == np.sign(op["mean_diff"])
        stab.append({"lang": lang, "sys_a": a, "sys_b": b,
                     "stable": bool(lw["sig"] and op["sig"] and same_sign), "sign_flip": bool(not same_sign)})
    return df.merge(pd.DataFrame(stab), on=["lang", "sys_a", "sys_b"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--langs", nargs="+", choices=langs(), help="default: every configured language with runs")
    args = parser.parse_args(argv)
    ql = query_level(args.langs)
    ql.to_csv(path("results") / "bounds_query_level.csv", index=False)
    summ = summarise(ql)
    summ.to_csv(path("results") / "bounds.csv", index=False)
    viol = violations(ql)
    viol.to_csv(path("results") / "optimistic_violations.csv", index=False)
    g = gaps(ql)
    g.to_csv(path("results") / "gaps.csv", index=False)
    p = pairs(ql)
    p.to_csv(path("results") / "pairs.csv", index=False)
    print(summ.to_string(index=False))
    print(g.to_string(index=False) if not g.empty else "[bounds] no gaps computed")
    if not p.empty:
        print(p[p.treatment == "lower"][["lang", "sys_a", "sys_b", "mean_diff", "p_holm", "stable", "sign_flip"]].to_string(index=False))
    print(f"[bounds] optimistic<lower violations: {len(viol)}")
    print(f"[bounds] all_relevant<lower queries: {len(all_relevant_below_lower(ql))} of {len(ql)}")


if __name__ == "__main__":
    main()
