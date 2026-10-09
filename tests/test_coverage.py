import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("coverage_mod", Path("src/04_coverage.py"))
cov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cov)


@pytest.fixture(autouse=True)
def few_resamples(monkeypatch):
    monkeypatch.setitem(cov.cfg(), "bootstrap_resamples", 200)


def _frame():
    """yo (low, dense) and bn (mid, dense), bm25 contributor, mcontriever not."""
    rows = []
    judged = {("yo", "bm25"): 0.9, ("yo", "mcontriever"): 0.5,
              ("bn", "bm25"): 0.8, ("bn", "mcontriever"): 0.6}
    for (lang, system), base in judged.items():
        for i in range(20):
            rows.append({"lang": lang, "tier": cov.tier(lang), "system": system,
                         "contributor": cov.cfg()["systems"][system]["contributor"],
                         "qid": f"q{i}", "has_positive": i % 5 != 0,
                         "judged10": base + (0.1 if i % 2 else -0.1)})
    return pd.DataFrame(rows)


def test_summarise_one_row_per_cell_with_ordered_cis():
    s = cov.summarise(_frame())
    assert len(s) == 4
    assert (s.ci_lo <= s.judged10_mean).all() and (s.judged10_mean <= s.ci_hi).all()
    assert (s.ci_lo_pos <= s.judged10_mean_pos).all() and (s.judged10_mean_pos <= s.ci_hi_pos).all()
    assert (s.n_queries == 20).all() and (s.n_queries_pos == 16).all()
    row = s[(s.lang == "yo") & (s.system == "bm25")].iloc[0]
    assert row.judged10_mean == pytest.approx(0.9, abs=1e-4)


def test_breakdown_contains_ols_line_and_table():
    text = cov.breakdown(_frame())
    assert "intercept=" in text and "low_tier=" in text and "contributor=" in text and "interaction=" in text
    assert "tier x contributor" in text


def test_breakdown_ols_recovers_contributor_effect():
    text = cov.breakdown(_frame())
    line = next(l for l in text.splitlines() if l.startswith("intercept="))
    coefs = dict(part.split("=") for part in line.split())
    assert float(coefs["contributor"]) == pytest.approx(0.2, abs=1e-4)
    assert float(coefs["interaction"]) == pytest.approx(0.2, abs=1e-4)


def test_breakdown_h3_contrast_only_when_both_systems_present():
    f = _frame()
    assert "yo:" not in cov.breakdown(f)
    e5 = f[(f.lang == "yo") & (f.system == "mcontriever")].assign(system="e5small", judged10=0.7)
    text = cov.breakdown(pd.concat([f, e5], ignore_index=True))
    assert "yo: diff=" in text


def _write_lang(root, lang, systems):
    (root / "qrels").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "qrels" / f"miracl.{lang}.dev.txt").write_text("q1 0 d1 1\nq1 0 d2 0\nq2 0 d9 0\n")
    for s in systems:
        (root / "runs" / f"miracl.{lang}.{s}.txt").write_text(
            "q1 Q0 d1 1 2.0 x\nq1 Q0 d3 2 1.0 x\nq2 Q0 d4 1 1.0 x\n")


def test_query_level_reads_runs_and_skips_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cov, "path", lambda key: tmp_path / key)
    _write_lang(tmp_path, "yo", ["bm25"])
    ql = cov.query_level(["yo", "sw"])
    assert set(ql.system) == {"bm25"} and set(ql.lang) == {"yo"}
    q1 = ql[ql.qid == "q1"].iloc[0]
    assert q1.judged10 == pytest.approx(0.1) and bool(q1.has_positive)
    q2 = ql[ql.qid == "q2"].iloc[0]
    assert q2.judged10 == 0.0 and not bool(q2.has_positive)


def test_selected_langs_drops_languages_without_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(cov, "path", lambda key: tmp_path / key)
    _write_lang(tmp_path, "yo", ["bm25"])
    assert cov.selected_langs(None) == ["yo"]
    assert cov.selected_langs(["sw", "yo"]) == ["yo"]
