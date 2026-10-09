"""tables, summary numbers and minimum detectable effects from the result csvs.

reads only the result csvs. when an optional csv is missing its columns stay blank and its
table is not written.

outputs
  results/table1.csv, table1_gaps.csv, table2.csv, mde.csv, summary_stats.txt
  results/tables/table1.tex, gaps.tex, table2.tex, ciral.tex   (table rows only, no tabular)
  results/numbers.md                                            (each number with its source csv)
"""
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats as st

from src.utils.config import ROOT, cfg, langs

TABLE1_COLS = ["lang", "tier", "system", "contributor", "judged10", "ndcg_lower", "ndcg_condensed",
               "ndcg_optimistic", "width", "ndcg_corrected"]
TABLE2_COLS = ["lang", "model", "kappa", "precision", "recall", "accuracy", "unparseable_rate",
               "sec_per_judgement", "passes_gate"]
GAP_TREATMENTS = ["lower", "optimistic", "corrected"]
CIRAL_COLS = ["lang", "system", "judged10_shallow_mean", "judged10_deep_mean", "ndcg10_shallow_mean", "ndcg10_deep_mean"]

LOW_POWER_N = 200

TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def fmt_ci(m: float, lo: float, hi: float) -> str:
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_num(v, digits: int = 3) -> str:
    """number as text, blank for missing."""
    if v is None or pd.isna(v):
        return ""
    return f"{v:.{digits}f}"


def yes_no(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return "yes" if bool(v) else "no"


def tex_escape(text) -> str:
    out = "".join(TEX_SPECIAL.get(ch, ch) for ch in str(text))
    return re.sub(r"(?<!\w)-(?=\d)", "$-$", out)


def tex_line(fields) -> str:
    return " & ".join(tex_escape(f) for f in fields) + r" \\"


def mde_paired(diffs: pd.Series, alpha: float = 0.05, power: float = 0.8) -> float:
    """smallest mean paired difference a two sided test detects at the observed sd."""
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return float(z * diffs.std(ddof=1) / math.sqrt(len(diffs)))


def mde_table(ql: pd.DataFrame, column: str = "lower") -> pd.DataFrame:
    """minimum detectable paired ndcg@10 difference per language and system pair."""
    order = system_order(ql.system.unique())
    rows = []
    for lang, g in ql.groupby("lang", sort=False):
        wide = g.pivot(index="qid", columns="system", values=column)
        systems = [s for s in order if s in wide.columns]
        for a, b in combinations(systems, 2):
            d = (wide[a] - wide[b]).dropna()
            rows.append({"lang": lang, "n_queries": len(d), "sys_a": a, "sys_b": b,
                         "sd_diff": round(float(d.std(ddof=1)), 4), "mde_ndcg10": round(mde_paired(d), 4),
                         "low_power": len(d) < LOW_POWER_N})
    return pd.DataFrame(rows)


def system_order(present) -> list[str]:
    known = [s for s in cfg()["systems"] if s in set(present)]
    return known + [s for s in present if s not in known]


def sort_cells(df: pd.DataFrame) -> pd.DataFrame:
    """config order for languages and systems, unknown ones last."""
    lang_pos = {l: i for i, l in enumerate(langs())}
    sys_pos = {s: i for i, s in enumerate(cfg()["systems"])}
    key = pd.DataFrame({"l": df.lang.map(lang_pos).fillna(len(lang_pos)), "s": df.system.map(sys_pos).fillna(len(sys_pos))},
                       index=df.index)
    return df.loc[key.sort_values(["l", "s"], kind="stable").index].reset_index(drop=True)


def table1(cov: pd.DataFrame, bnd: pd.DataFrame, filled: pd.DataFrame | None) -> pd.DataFrame:
    df = cov.merge(bnd, on=["lang", "system"], suffixes=("", "_b"))
    df = sort_cells(df)
    df["judged10"] = [fmt_ci(m, lo, hi) for m, lo, hi in zip(df.judged10_mean, df.ci_lo, df.ci_hi)]
    df["ndcg_lower"] = df.lower_mean.map(fmt_num)
    df["ndcg_condensed"] = df.condensed_mean.map(fmt_num)
    df["ndcg_optimistic"] = df.optimistic_mean.map(fmt_num)
    df["width"] = df.width.map(fmt_num)
    df["ndcg_corrected"] = ""
    if filled is not None:
        corrected = filled.set_index(["lang", "system"]).corrected_mean
        df["ndcg_corrected"] = [fmt_num(corrected.get((l, s))) for l, s in zip(df.lang, df.system)]
    return df[TABLE1_COLS]


def table1_tex(t1: pd.DataFrame) -> str:
    """width stays in table1.csv, the tex table has no column for it."""
    rows = t1.drop(columns="width").assign(contributor=t1.contributor.map(yes_no))
    return "\n".join(tex_line(r) for r in rows.itertuples(index=False))


def gaps_table(gaps: pd.DataFrame, gaps_filled: pd.DataFrame | None) -> pd.DataFrame:
    """every gap and treatment with ci text. rows from gaps_filled are on the llm query sample."""
    out = gaps.assign(scope="full")
    if gaps_filled is not None:
        out = pd.concat([out, gaps_filled.assign(scope="sample")], ignore_index=True)
    out["ci"] = [fmt_ci(d, lo, hi) for d, lo, hi in zip(out.delta, out.ci_lo, out.ci_hi)]
    return out


def gaps_wide(g: pd.DataFrame) -> pd.DataFrame:
    """one row per gap and scope, the treatments side by side."""
    keys = ["system", "gap", "scope"]
    first = g.drop_duplicates(keys).set_index(keys)
    wide = g.pivot(index=keys, columns="treatment", values="ci").reindex(index=first.index, columns=GAP_TREATMENTS)
    return wide.assign(identified=first.identified).reset_index()


def gaps_tex(g: pd.DataFrame) -> str:
    w = gaps_wide(g)
    w["gap"] = w.gap.where(w.scope == "full", w.gap + " (sample)")
    w["identified"] = w.identified.map(yes_no)
    rows = w[["system", "gap", *GAP_TREATMENTS, "identified"]].fillna("")
    return "\n".join(tex_line(r) for r in rows.itertuples(index=False))


def table2(val: pd.DataFrame | None) -> pd.DataFrame | None:
    return None if val is None else val.reindex(columns=TABLE2_COLS)


def table2_tex(t2: pd.DataFrame) -> str:
    lines = []
    for r in t2.itertuples(index=False):
        nums = [fmt_num(v) for v in (r.kappa, r.precision, r.recall, r.accuracy, r.unparseable_rate)]
        lines.append(tex_line([r.lang, r.model, *nums, fmt_num(r.sec_per_judgement, 2), yes_no(r.passes_gate)]))
    return "\n".join(lines)


def ciral_table(c: pd.DataFrame | None) -> pd.DataFrame | None:
    """rows for the systems run in every language, the rest stay in ciral.csv."""
    if c is None:
        return None
    everywhere = c.groupby("system").lang.nunique() == c.lang.nunique()
    return c[c.system.isin(everywhere[everywhere].index)].reindex(columns=CIRAL_COLS)


def ciral_tex(t: pd.DataFrame) -> str:
    return "\n".join(tex_line([r.lang, r.system, *(fmt_num(v) for v in r[2:])]) for r in t.itertuples(index=False))


@dataclass
class Fact:
    key: str
    value: str
    source: str
    columns: str
    rule: str


def share(k: int, n: int) -> str:
    return f"{k} of {n} ({100 * k / n:.0f}%)" if n else f"{k} of 0"


def span(v: pd.Series) -> str:
    return f"{v.min():.2f} to {v.max():.2f}"


def cell_range(df: pd.DataFrame, col: str) -> str:
    lo, hi = df.loc[df[col].idxmin()], df.loc[df[col].idxmax()]
    return f"{lo[col]:.3f} ({lo.lang} {lo.system}) to {hi[col]:.3f} ({hi.lang} {hi.system})"


def unique_pairs(pairs: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    """one row per pair. ciral pairs come per metric, nDCG@10 is the one reported."""
    if "metric" in pairs.columns:
        pairs = pairs[pairs.metric == "ndcg10"]
    return pairs.drop_duplicates(["lang", "sys_a", "sys_b"])[flags]


def summary_facts(frames: dict) -> list[Fact]:
    """headline numbers. a fact is only made when its csv was given."""
    f = {k: v for k, v in frames.items() if v is not None}
    facts = []
    if "coverage" in f:
        cov = f["coverage"]
        contrib = cov[cov.contributor.astype(bool)].judged10_mean.mean()
        other = cov[~cov.contributor.astype(bool)].judged10_mean.mean()
        facts.append(Fact("judged10_range", cell_range(cov, "judged10_mean"), "results/coverage.csv",
                          "judged10_mean, lang, system", "min and max over lang x system cells"))
        facts.append(Fact("judged10_by_contributor", f"{contrib:.3f} contributors versus {other:.3f} non-contributors",
                          "results/coverage.csv", "judged10_mean, contributor",
                          "mean of the cell means, grouped by contributor"))
    if "bounds" in f:
        facts.append(Fact("width_range", cell_range(f["bounds"], "width"), "results/bounds.csv",
                          "width, lang, system", "min and max over lang x system cells"))
    if "gaps" in f:
        g = f["gaps"].dropna(subset=["identified"]).drop_duplicates(["system", "gap"])
        tier = g.gap.str.startswith("tier:")
        for key, part in [("gaps_identified", g), ("gaps_identified_tier", g[tier]), ("gaps_identified_pairwise", g[~tier])]:
            facts.append(Fact(key, share(int(part.identified.astype(bool).sum()), len(part)), "results/gaps.csv",
                              "system, gap, identified", "one row per system and gap, tier gaps start with 'tier:'"))
    if "pairs" in f:
        p = f["pairs"]
        u = unique_pairs(p, ["stable", "sign_flip"])
        sig_lower = p[(p.treatment == "lower") & p.sig.astype(bool)].drop_duplicates(["lang", "sys_a", "sys_b"])
        facts.append(Fact("pairs_stable", share(int(u.stable.sum()), len(u)), "results/pairs.csv",
                          "lang, sys_a, sys_b, stable", "one row per language and system pair"))
        facts.append(Fact("pairs_sign_flip", share(int(u.sign_flip.sum()), len(u)), "results/pairs.csv",
                          "lang, sys_a, sys_b, sign_flip", "one row per language and system pair"))
        facts.append(Fact("pairs_stable_of_sig_lower", share(int(sig_lower.stable.sum()), len(sig_lower)), "results/pairs.csv",
                          "treatment, sig, stable", "pairs significant under lower that are also stable"))
    if "data_audit" in f and "te" in set(f["data_audit"].lang):
        a = f["data_audit"].set_index("lang").judgements_per_query_mean
        rest = a.drop("te")
        facts.append(Fact("density_te", f"te {a['te']:.2f} judgements per query versus {span(rest)} in the other {len(rest)} languages",
                          "results/data_audit.csv", "judgements_per_query_mean", "te against min and max of the rest"))
    if "mde" in f and len(f["mde"]):
        m = f["mde"]
        low = sorted(m[m.low_power.astype(bool)].lang.unique())
        facts.append(Fact("mde_range", f"{m.mde_ndcg10.min():.3f} to {m.mde_ndcg10.max():.3f}", "results/mde.csv",
                          "mde_ndcg10", "min and max over language x system pair"))
        facts.append(Fact("mde_low_power_langs", ", ".join(low) if low else "none", "results/mde.csv",
                          "lang, low_power", f"languages with fewer than {LOW_POWER_N} queries"))
    if "llm_validation" in f:
        v = f["llm_validation"]
        facts.append(Fact("llm_kappa_range", span(v.kappa), "results/llm_validation.csv", "kappa", "min and max over languages"))
        facts.append(Fact("llm_gate_passed", share(int(v.passes_gate.astype(bool).sum()), len(v)), "results/llm_validation.csv",
                          "passes_gate", "languages that pass the agreement gate"))
    if "filled" in f:
        fl = f["filled"]
        facts.append(Fact("holes_per_query_range", span(fl.holes_per_query), "results/filled.csv", "holes_per_query",
                          "min and max over lang x system cells"))
        facts.append(Fact("llm_rel_rate_range", span(fl.llm_rel_rate), "results/filled.csv", "llm_rel_rate",
                          "min and max over lang x system cells"))
    if "pairs_filled" in f:
        u = unique_pairs(f["pairs_filled"], ["stable"])
        facts.append(Fact("pairs_filled_stable", share(int(u.stable.sum()), len(u)), "results/pairs_filled.csv",
                          "lang, sys_a, sys_b, stable", "stable between lower and corrected on the llm sample"))
    if "ciral_pairs" in f:
        u = unique_pairs(f["ciral_pairs"], ["sign_flip", "sig_change"])
        facts.append(Fact("ciral_sign_flips", share(int(u.sign_flip.sum()), len(u)), "results/ciral_pairs.csv",
                          "lang, sys_a, sys_b, sign_flip", "shallow against deep qrels"))
        facts.append(Fact("ciral_sig_changes", share(int(u.sig_change.sum()), len(u)), "results/ciral_pairs.csv",
                          "lang, sys_a, sys_b, sig_change", "shallow against deep qrels"))
    return facts


def summary_text(facts: list[Fact]) -> str:
    return "\n".join(f"{x.key}: {x.value}" for x in facts) + "\n"


def numbers_md(facts: list[Fact]) -> str:
    head = ["# numbers from src/09_stats.py", "",
            "every value below is computed from the listed csv.", "",
            "| key | value | source | columns | rule |", "|---|---|---|---|---|"]
    rows = [f"| {x.key} | {x.value} | {x.source} | {x.columns} | {x.rule} |" for x in facts]
    return "\n".join(head + rows) + "\n"


def read_csv(results: Path, name: str, required: bool = False) -> pd.DataFrame | None:
    file = results / name
    if file.exists():
        return pd.read_csv(file)
    if required:
        raise FileNotFoundError(file)
    return None


def write_or_remove(file: Path, text: str | None) -> None:
    """write text, or delete a stale file from an earlier run when there is nothing to write."""
    if text is None:
        file.unlink(missing_ok=True)
    else:
        file.write_text(text + "\n", encoding="utf-8", newline="\n")


def build(results: Path) -> pd.DataFrame:
    cov = read_csv(results, "coverage.csv", required=True)
    bnd = read_csv(results, "bounds.csv", required=True)
    gaps = read_csv(results, "gaps.csv", required=True)
    pairs = read_csv(results, "pairs.csv", required=True)
    audit = read_csv(results, "data_audit.csv", required=True)
    ql = read_csv(results, "bounds_query_level.csv", required=True)
    filled = read_csv(results, "filled.csv")
    gaps_filled = read_csv(results, "gaps_filled.csv")
    val = read_csv(results, "llm_validation.csv")
    ciral = read_csv(results, "ciral.csv")
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    t1 = table1(cov, bnd, filled)
    t1.to_csv(results / "table1.csv", index=False)
    write_or_remove(tables / "table1.tex", table1_tex(t1))

    g = gaps_table(gaps, gaps_filled)
    g.to_csv(results / "table1_gaps.csv", index=False)
    write_or_remove(tables / "gaps.tex", gaps_tex(g))

    t2 = table2(val)
    if t2 is None:
        (results / "table2.csv").unlink(missing_ok=True)
    else:
        t2.to_csv(results / "table2.csv", index=False)
    write_or_remove(tables / "table2.tex", None if t2 is None else table2_tex(t2))

    ct = ciral_table(ciral)
    write_or_remove(tables / "ciral.tex", None if ct is None else ciral_tex(ct))

    mde = mde_table(ql)
    mde.to_csv(results / "mde.csv", index=False)

    frames = {"coverage": cov, "bounds": bnd, "gaps": gaps, "pairs": pairs, "data_audit": audit, "mde": mde,
              "llm_validation": val, "filled": filled, "pairs_filled": read_csv(results, "pairs_filled.csv"),
              "ciral_pairs": read_csv(results, "ciral_pairs.csv")}
    facts = summary_facts(frames)
    (results / "summary_stats.txt").write_text(summary_text(facts), encoding="utf-8")
    (results / "numbers.md").write_text(numbers_md(facts), encoding="utf-8")
    return t1


def main() -> None:
    t1 = build(ROOT / "results")
    print(t1.to_string(index=False))
    print((ROOT / "results" / "mde.csv").read_text(encoding="utf-8"))
    print((ROOT / "results" / "summary_stats.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
