"""judged@10 per language and system with bootstrap cis, plus a tier by contributor breakdown.

reads runs/miracl.{lang}.{system}.txt and the dev qrels, writes three files in results/.
languages or systems without a run file are skipped.
"""
import argparse

import numpy as np
import pandas as pd

from src.utils.bootstrap import bootstrap_ci
from src.utils.config import cfg, langs, path, tier
from src.utils.io import read_qrels, read_run
from src.utils.metrics import judged_at_k


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
                ranking = run.get(qid, [])
                rows.append({"lang": lang, "tier": tier(lang), "system": system,
                             "contributor": cfg()["systems"][system]["contributor"], "qid": qid,
                             "has_positive": any(r > 0 for r in rel.values()),
                             "judged10": judged_at_k(ranking, rel, k)})
    return pd.DataFrame(rows)


def summarise(ql: pd.DataFrame) -> pd.DataFrame:
    n, seed = cfg()["bootstrap_resamples"], cfg()["seed"]
    rows = []
    for (lang, system), g in ql.groupby(["lang", "system"], sort=False):
        mean, lo, hi = bootstrap_ci(g.judged10.values, n, seed)
        gp = g[g.has_positive]
        mean_p, lo_p, hi_p = bootstrap_ci(gp.judged10.values, n, seed)
        rows.append({"lang": lang, "tier": tier(lang), "system": system,
                     "contributor": cfg()["systems"][system]["contributor"],
                     "n_queries": len(g), "judged10_mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "n_queries_pos": len(gp), "judged10_mean_pos": round(mean_p, 4),
                     "ci_lo_pos": round(lo_p, 4), "ci_hi_pos": round(hi_p, 4)})
    return pd.DataFrame(rows)


def breakdown(ql: pd.DataFrame) -> str:
    """two-way table and a tiny ols on dense languages only (where contributor status varies)."""
    dense = ql[ql.lang.isin(langs(dense_only=True))]
    lines = ["# Judged@10 by tier x contributor (dense languages only; mean over queries)",
             dense.pivot_table(index="tier", columns="contributor", values="judged10", aggfunc="mean").round(4).to_string(), ""]
    # judged10 ~ low_tier + contributor + low_tier*contributor
    low = (dense.tier == "low").to_numpy(dtype=float)
    contributor = dense.contributor.to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(dense)), low, contributor, low * contributor])
    beta, *_ = np.linalg.lstsq(X, dense.judged10.to_numpy(dtype=float), rcond=None)
    lines += ["# OLS judged10 ~ 1 + low_tier + contributor + low_tier:contributor",
              f"intercept={beta[0]:.4f} low_tier={beta[1]:.4f} contributor={beta[2]:.4f} interaction={beta[3]:.4f}",
              "a language effect predicts low_tier < 0 for both contributor levels;",
              "a contributor bias predicts contributor > 0 regardless of tier.", ""]
    # e5small against mcontriever on the same queries, per e5 language
    lines.append("# contrast: e5small - mcontriever Judged@10 (paired over queries)")
    for lang in cfg()["e5_languages"]:
        a = ql[(ql.lang == lang) & (ql.system == "e5small")].set_index("qid").judged10
        b = ql[(ql.lang == lang) & (ql.system == "mcontriever")].set_index("qid").judged10
        if len(a) and len(b):
            d = (a - b.reindex(a.index)).dropna().to_numpy()
            mean, lo, hi = bootstrap_ci(d, cfg()["bootstrap_resamples"], cfg()["seed"])
            lines.append(f"{lang}: diff={mean:.4f} CI=[{lo:.4f}, {hi:.4f}] n={len(d)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--langs", nargs="+", choices=langs(), help="default: every configured language with runs")
    args = parser.parse_args(argv)
    ql = query_level(args.langs)
    ql.to_csv(path("results") / "coverage_query_level.csv", index=False)
    summ = summarise(ql)
    summ.to_csv(path("results") / "coverage.csv", index=False)
    (path("results") / "coverage_model.txt").write_text(breakdown(ql), encoding="utf-8")
    print(summ.to_string(index=False))
    print((path("results") / "coverage_model.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
