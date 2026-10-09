import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("bounds_mod", Path("src/05_bounds.py"))
bounds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounds)


@pytest.fixture(autouse=True)
def small_counts(monkeypatch):
    monkeypatch.setitem(bounds.cfg(), "bootstrap_resamples", 300)
    monkeypatch.setitem(bounds.cfg(), "permutations", 300)


def _cell(lang, system, lower, optimistic, n=30, seed=0):
    """n queries with noise around given means; condensed sits between."""
    noise = np.random.default_rng(seed).uniform(-0.05, 0.05, size=n)
    lo = np.clip(lower + noise, 0, 1)
    op = np.clip(optimistic + noise, 0, 1)
    return pd.DataFrame({"lang": lang, "tier": bounds.tier(lang), "system": system,
                         "qid": [f"q{i}" for i in range(n)], "has_positive": True,
                         "lower": lo, "condensed": (lo + op) / 2, "optimistic": op,
                         "all_relevant": op, "optimistic_fixed": op})


def test_summarise_width_and_ordering():
    ql = pd.concat([_cell("yo", "bm25", 0.4, 0.6), _cell("bn", "mcontriever", 0.5, 0.9, seed=1)], ignore_index=True)
    s = bounds.summarise(ql)
    assert len(s) == 2
    assert (s.lower_mean <= s.optimistic_mean).all()
    for _, r in s.iterrows():
        assert r.width == pytest.approx(r.optimistic_mean - r.lower_mean, abs=1e-9)
        for t in bounds.TREATMENTS:
            assert r[f"{t}_lo"] <= r[f"{t}_mean"] <= r[f"{t}_hi"]
    assert s.set_index("system").loc["mcontriever", "width"] == pytest.approx(0.4, abs=0.01)


def test_violations_empty_for_sane_data_and_flags_bad_rows():
    ql = _cell("yo", "bm25", 0.4, 0.6)
    assert bounds.violations(ql).empty
    bad = ql.copy()
    bad.loc[0, "optimistic"] = bad.loc[0, "lower"] - 0.2
    v = bounds.violations(bad)
    assert list(v.qid) == ["q0"]


def test_all_relevant_below_lower_is_counted_separately():
    ql = _cell("yo", "bm25", 0.4, 0.6)
    assert bounds.all_relevant_below_lower(ql).empty
    low = ql.copy()
    low.loc[[0, 1], "all_relevant"] = low.loc[[0, 1], "lower"] - 0.1
    assert list(bounds.all_relevant_below_lower(low).qid) == ["q0", "q1"]
    assert bounds.violations(low).empty


def _gap_frame():
    """bm25: bn clearly above yo under both treatments. mdpr: gap vanishes under optimistic.
    mcontriever: gap flips sign."""
    parts = [_cell("bn", "bm25", 0.8, 0.9, seed=1), _cell("yo", "bm25", 0.2, 0.4, seed=2)]
    same = _cell("yo", "mdpr", 0.5, 0.7, seed=3)
    parts += [_cell("bn", "mdpr", 0.8, 0.7, seed=3).assign(optimistic=same.optimistic.values), same]
    parts += [_cell("bn", "mcontriever", 0.8, 0.3, seed=4), _cell("yo", "mcontriever", 0.2, 0.9, seed=5)]
    return pd.concat(parts, ignore_index=True)


def _identified(g, system):
    return g[(g.system == system) & (g.gap == "bn - yo")].identified.unique().tolist()


def test_gaps_identified_only_when_both_treatments_agree():
    g = bounds.gaps(_gap_frame())
    assert _identified(g, "bm25") == [True]
    assert _identified(g, "mdpr") == [False]
    assert _identified(g, "mcontriever") == [False]
    assert set(g.treatment) == {"lower", "condensed", "optimistic"}
    assert (g.ci_lo <= g.delta).all() and (g.delta <= g.ci_hi).all()


def test_gaps_includes_tier_level_row():
    g = bounds.gaps(_gap_frame())
    tier_rows = g[(g.system == "bm25") & g.gap.str.startswith("tier:")]
    assert set(tier_rows.gap) == {"tier:bn - yo"}
    assert len(tier_rows) == 3


def test_gaps_respects_treatments_argument():
    g = bounds.gaps(_gap_frame(), treatments=("lower", "optimistic"))
    assert set(g.treatment) == {"lower", "optimistic"}


def _pair_frame():
    """yo: bm25 clearly better under both treatments. sw: sign flips between treatments.
    bn: no real difference."""
    n = 30
    qids = [f"q{i}" for i in range(n)]
    wobble = np.where(np.arange(n) % 2 == 0, 0.01, -0.01)
    cells = {
        ("yo", "bm25"): (0.8 + wobble, 0.9 + wobble), ("yo", "mdpr"): (0.3 + wobble, 0.4 + wobble),
        ("sw", "bm25"): (0.8 + wobble, 0.3 + wobble), ("sw", "mdpr"): (0.3 + wobble, 0.8 + wobble),
        ("bn", "bm25"): (0.5 + wobble, 0.5 + wobble), ("bn", "mdpr"): (0.5 - wobble, 0.5 - wobble),
    }
    rows = [pd.DataFrame({"lang": lang, "system": system, "qid": qids, "lower": lo, "optimistic": op})
            for (lang, system), (lo, op) in cells.items()]
    return pd.concat(rows, ignore_index=True)


def _pair_row(p, lang):
    return p[(p.lang == lang) & (p.treatment == "lower")].iloc[0]


def test_pairs_stable_and_sign_flip():
    p = bounds.pairs(_pair_frame())
    yo, sw, bn = _pair_row(p, "yo"), _pair_row(p, "sw"), _pair_row(p, "bn")
    assert bool(yo.stable) and not bool(yo.sign_flip)
    assert not bool(sw.stable) and bool(sw.sign_flip)
    assert not bool(bn.stable) and not bool(bn.sign_flip)
    assert set(p.treatment) == {"lower", "optimistic"}


def test_pairs_holm_adjusted_p_not_below_raw():
    p = bounds.pairs(_pair_frame())
    assert (p.p_holm >= p.p_raw - 1e-4).all()


def _write_lang(root, lang, systems):
    (root / "qrels").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "qrels" / f"miracl.{lang}.dev.txt").write_text("q1 0 d1 1\nq1 0 d2 0\nq2 0 d9 0\n")
    for s in systems:
        (root / "runs" / f"miracl.{lang}.{s}.txt").write_text(
            "q1 Q0 d1 1 2.0 x\nq1 Q0 d3 2 1.0 x\nq2 Q0 d4 1 1.0 x\n")


def test_query_level_scores_all_treatments_and_skips_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bounds, "path", lambda key: tmp_path / key)
    _write_lang(tmp_path, "yo", ["bm25"])
    ql = bounds.query_level(["yo", "sw"])
    assert set(ql.lang) == {"yo"} and set(ql.system) == {"bm25"}
    q1 = ql[ql.qid == "q1"].iloc[0]
    assert q1.lower == pytest.approx(1.0)
    assert q1.optimistic >= q1.lower and q1.optimistic <= 1.0 + 1e-9
    assert {"all_relevant", "optimistic_fixed"} <= set(ql.columns)
    q2 = ql[ql.qid == "q2"].iloc[0]
    assert q2.lower == 0.0 and not bool(q2.has_positive)
    assert bounds.violations(ql).empty


def test_selected_langs_drops_languages_without_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(bounds, "path", lambda key: tmp_path / key)
    _write_lang(tmp_path, "yo", ["bm25"])
    assert bounds.selected_langs(None) == ["yo"]
    assert bounds.selected_langs(["sw", "yo"]) == ["yo"]
