import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("stats_mod", Path("src/09_stats.py"))
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def _cov():
    rows = []
    for lang, tier in [("yo", "low"), ("fi", "high")]:
        for system, contrib, j in [("bm25", True, 0.9), ("mcontriever", False, 0.5)]:
            rows.append({"lang": lang, "tier": tier, "system": system, "contributor": contrib,
                         "n_queries": 20, "judged10_mean": j, "ci_lo": j - 0.05, "ci_hi": j + 0.05})
    return pd.DataFrame(rows)


def _bnd():
    rows = []
    for lang, tier in [("yo", "low"), ("fi", "high")]:
        for system, contrib, lo in [("bm25", True, 0.4), ("mcontriever", False, 0.3)]:
            rows.append({"lang": lang, "tier": tier, "system": system, "contributor": contrib, "n_queries": 20,
                         "lower_mean": lo, "condensed_mean": lo + 0.1, "optimistic_mean": lo + 0.3, "width": 0.3})
    return pd.DataFrame(rows)


def _gaps():
    rows = []
    for gap, ident in [("tier:fi - yo", True), ("fi - sw", False)]:
        for t in ["lower", "condensed", "optimistic"]:
            rows.append({"system": "bm25", "gap": gap, "treatment": t, "delta": 0.1, "ci_lo": 0.05, "ci_hi": 0.15,
                         "excludes_zero": True, "sign": 1.0, "identified": ident})
    return pd.DataFrame(rows)


def _pairs():
    rows = []
    for lang, stable, flip in [("yo", False, True), ("fi", True, False), ("fi", False, False)]:
        for t in ["lower", "optimistic"]:
            rows.append({"lang": lang, "treatment": t, "sys_a": "bm25", "sys_b": "mdpr" if stable else "mcontriever",
                         "mean_diff": 0.1, "p_raw": 0.01, "p_holm": 0.02, "sig": stable, "stable": stable, "sign_flip": flip})
    return pd.DataFrame(rows)


def _audit():
    return pd.DataFrame({"lang": ["yo", "te", "fi"], "tier": ["low", "mid", "high"],
                         "judgements_per_query_mean": [9.98, 1.94, 9.45], "corpus_passages": [49043, 518079, 1883509]})


def _query_level(n=30, systems=("bm25", "mdpr"), langs=("yo",), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for lang in langs:
        for q in range(n):
            base = rng.uniform(0.2, 0.8)
            for s in systems:
                rows.append({"lang": lang, "system": s, "qid": f"q{q}", "lower": float(np.clip(base + rng.normal(0, 0.15), 0, 1))})
    return pd.DataFrame(rows)


def test_fmt_ci():
    assert stats.fmt_ci(0.5123, 0.4011, 0.6199) == "0.512 [0.401, 0.620]"


def test_fmt_num_blank_for_nan():
    assert stats.fmt_num(0.12345) == "0.123"
    assert stats.fmt_num(float("nan")) == ""
    assert stats.fmt_num(None) == ""


def test_tex_escape_special_characters():
    assert stats.tex_escape("a_b") == r"a\_b"
    assert stats.tex_escape("50% & #1") == r"50\% \& \#1"
    assert stats.tex_escape("x - y") == "x - y"


def test_tex_escape_minus_before_digit():
    assert stats.tex_escape("-0.038 [-0.1, 0.2]") == "$-$0.038 [$-$0.1, 0.2]"


def test_mde_paired_scales_with_sqrt_n():
    d_small = pd.Series([0.1, -0.1] * 10)
    d_big = pd.Series([0.1, -0.1] * 40)
    assert stats.mde_paired(d_small) > stats.mde_paired(d_big)
    assert abs(stats.mde_paired(d_small) / stats.mde_paired(d_big) - 2.0) < 0.05


def test_mde_table_pairs_and_low_power_flag():
    ql = pd.concat([_query_level(n=30, langs=("yo",)), _query_level(n=300, langs=("fi",), seed=1)])
    m = stats.mde_table(ql)
    assert set(m.lang) == {"yo", "fi"}
    assert len(m) == 2
    assert m.set_index("lang").low_power.to_dict() == {"yo": True, "fi": False}
    assert (m.mde_ndcg10 > 0).all()


def test_table1_merges_coverage_and_bounds():
    t1 = stats.table1(_cov(), _bnd(), filled=None)
    assert list(t1.columns) == stats.TABLE1_COLS
    assert len(t1) == 4
    assert t1.ndcg_corrected.eq("").all()
    assert t1.judged10.iloc[0] == "0.900 [0.850, 0.950]"


def test_table1_follows_config_language_order():
    t1 = stats.table1(_cov().iloc[::-1], _bnd(), filled=None)
    assert list(t1.lang) == ["yo", "yo", "fi", "fi"]
    assert list(t1.system[:2]) == ["bm25", "mcontriever"]


def test_table1_adds_corrected_when_available():
    filled = pd.DataFrame({"lang": ["yo"], "system": ["bm25"], "corrected_mean": [0.512]})
    t1 = stats.table1(_cov(), _bnd(), filled)
    assert t1.set_index(["lang", "system"]).ndcg_corrected.to_dict() == {
        ("yo", "bm25"): "0.512", ("yo", "mcontriever"): "", ("fi", "bm25"): "", ("fi", "mcontriever"): ""}


def test_table1_tex_has_nine_fields_per_row():
    t1 = stats.table1(_cov(), _bnd(), filled=None)
    rows = stats.table1_tex(t1).splitlines()
    assert len(rows) == 4
    for row in rows:
        assert row.endswith(r" \\")
        assert row.count(" & ") == 8
    assert "yes" in rows[0] and "no" in rows[1]


def test_gaps_table_marks_sample_rows():
    gf = _gaps().assign(treatment=["lower", "optimistic", "corrected"] * 2)
    g = stats.gaps_table(_gaps(), gf)
    assert len(g) == 12
    assert g.scope.tolist() == ["full"] * 6 + ["sample"] * 6
    assert g.ci.iloc[0] == "0.100 [0.050, 0.150]"


def test_gaps_table_without_filled():
    assert len(stats.gaps_table(_gaps(), None)) == 6


def test_gaps_wide_one_row_per_gap_and_scope():
    gf = _gaps().assign(treatment=["lower", "optimistic", "corrected"] * 2)
    w = stats.gaps_wide(stats.gaps_table(_gaps(), gf))
    assert w[["gap", "scope"]].values.tolist() == [["tier:fi - yo", "full"], ["fi - sw", "full"],
                                                   ["tier:fi - yo", "sample"], ["fi - sw", "sample"]]
    assert w.corrected.isna().tolist() == [True, True, False, False]
    assert w.identified.tolist() == [True, False, True, False]


def test_gaps_tex_rows_have_six_fields_and_mark_sample():
    gf = _gaps().assign(treatment=["lower", "optimistic", "corrected"] * 2)
    rows = stats.gaps_tex(stats.gaps_table(_gaps(), gf)).splitlines()
    assert len(rows) == 4
    for row in rows:
        assert row.count(" & ") == 5
    ci = "0.100 [0.050, 0.150]"
    assert rows[0] == f"bm25 & tier:fi - yo & {ci} & {ci} &  & yes " + r"\\"
    assert rows[3] == f"bm25 & fi - sw (sample) & {ci} & {ci} & {ci} & no " + r"\\"


def test_gaps_tex_without_corrected_has_blank_column():
    rows = stats.gaps_tex(stats.gaps_table(_gaps(), None)).splitlines()
    assert len(rows) == 2
    assert all(r.count(" & ") == 5 for r in rows)


def test_table2_columns_and_blank_without_csv():
    v = pd.DataFrame({"lang": ["sw"], "model": ["qwen2.5_7b"], "kappa": [0.31234], "precision": [0.7], "recall": [0.6],
                      "accuracy": [0.8], "unparseable_rate": [0.01], "sec_per_judgement": [1.234], "passes_gate": [True]})
    t2 = stats.table2(v)
    assert list(t2.columns) == stats.TABLE2_COLS
    row = stats.table2_tex(t2).splitlines()[0]
    assert row.count(" & ") == 8
    assert r"qwen2.5\_7b" in row and row.startswith("sw & ")
    assert row.split(" & ")[2] == "0.312"
    assert stats.table2(None) is None


def test_ciral_tex_has_six_fields():
    c = pd.DataFrame({"lang": ["hau"], "system": ["bm25dt"], "n_queries": [60],
                      "judged10_shallow_mean": [0.9], "judged10_deep_mean": [0.95],
                      "ndcg10_shallow_mean": [0.3], "ndcg10_deep_mean": [0.35]})
    rows = stats.ciral_tex(stats.ciral_table(c)).splitlines()
    assert rows == [r"hau & bm25dt & 0.900 & 0.950 & 0.300 & 0.350 \\"]


def test_ciral_table_keeps_only_systems_run_in_every_language():
    c = pd.DataFrame({"lang": ["hau", "hau", "swa"], "system": ["bm25dt", "mdpr", "bm25dt"],
                      "judged10_shallow_mean": [0.9, 0.1, 0.8], "judged10_deep_mean": [0.95, 0.2, 0.9],
                      "ndcg10_shallow_mean": [0.3, 0.1, 0.2], "ndcg10_deep_mean": [0.35, 0.1, 0.25]})
    assert list(stats.ciral_table(c).system) == ["bm25dt", "bm25dt"]


def test_summary_facts_counts_from_frames():
    facts = {f.key: f.value for f in stats.summary_facts(
        {"coverage": _cov(), "bounds": _bnd(), "gaps": _gaps(), "pairs": _pairs(), "data_audit": _audit()})}
    assert facts["gaps_identified"] == "1 of 2 (50%)"
    assert facts["gaps_identified_tier"] == "1 of 1 (100%)"
    assert facts["gaps_identified_pairwise"] == "0 of 1 (0%)"
    assert facts["pairs_stable"] == "1 of 3 (33%)"
    assert facts["pairs_sign_flip"] == "1 of 3 (33%)"
    assert facts["judged10_range"].startswith("0.500 (yo mcontriever) to 0.900 (yo bm25)")
    assert facts["width_range"].startswith("0.300")
    assert "te 1.94" in facts["density_te"]
    assert "9.45 to 9.98" in facts["density_te"]


def test_summary_facts_skip_missing_optional_inputs():
    keys = {f.key for f in stats.summary_facts({"coverage": _cov(), "bounds": _bnd()})}
    assert "judged10_range" in keys
    assert not keys & {"gaps_identified", "pairs_stable", "density_te", "llm_kappa_range", "ciral_sign_flips"}


def test_numbers_md_lists_source_and_columns():
    facts = stats.summary_facts({"coverage": _cov()})
    md = stats.numbers_md(facts)
    assert "results/coverage.csv" in md
    assert "judged10_mean" in md
    assert md.count("\n") >= len(facts)


def _write_core(res):
    _cov().to_csv(res / "coverage.csv", index=False)
    _bnd().to_csv(res / "bounds.csv", index=False)
    _gaps().to_csv(res / "gaps.csv", index=False)
    _pairs().to_csv(res / "pairs.csv", index=False)
    _audit().to_csv(res / "data_audit.csv", index=False)
    _query_level(langs=("yo", "fi")).to_csv(res / "bounds_query_level.csv", index=False)


def test_build_with_core_csvs_only(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write_core(res)
    stats.build(res)
    for name in ["table1.csv", "table1_gaps.csv", "mde.csv", "summary_stats.txt"]:
        assert (res / name).exists()
    assert (res / "tables" / "table1.tex").exists()
    assert (res / "tables" / "gaps.tex").exists()
    assert (res / "numbers.md").exists()
    assert not (res / "table2.csv").exists()
    assert not (res / "tables" / "table2.tex").exists()
    assert not (res / "tables" / "ciral.tex").exists()
    assert (res / "table1.csv").read_text().count("\n") == 5


def test_build_writes_optional_tables_when_csvs_exist(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write_core(res)
    pd.DataFrame({"lang": ["yo"], "model": ["m"], "kappa": [0.3], "precision": [0.7], "recall": [0.6], "accuracy": [0.8],
                  "unparseable_rate": [0.0], "sec_per_judgement": [1.0], "passes_gate": [True]}).to_csv(res / "llm_validation.csv", index=False)
    pd.DataFrame({"lang": ["yo"], "system": ["bm25"], "corrected_mean": [0.55], "holes_per_query": [4.0],
                  "llm_rel_rate": [0.3]}).to_csv(res / "filled.csv", index=False)
    pd.DataFrame({"lang": ["hau"], "system": ["bm25dt"], "n_queries": [60], "judged10_shallow_mean": [0.9],
                  "judged10_deep_mean": [0.95], "ndcg10_shallow_mean": [0.3], "ndcg10_deep_mean": [0.35]}).to_csv(res / "ciral.csv", index=False)
    stats.build(res)
    assert (res / "table2.csv").exists()
    assert (res / "tables" / "table2.tex").exists()
    assert (res / "tables" / "ciral.tex").exists()
    assert "0.550" in (res / "table1.csv").read_text()


def test_stale_optional_tex_is_removed_when_csv_disappears(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write_core(res)
    (res / "tables").mkdir(parents=True)
    (res / "tables" / "table2.tex").write_text("old\n")
    stats.build(res)
    assert not (res / "tables" / "table2.tex").exists()


def test_build_fails_loudly_without_core_csv(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    with pytest.raises(FileNotFoundError):
        stats.build(res)
