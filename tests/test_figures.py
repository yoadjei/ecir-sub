import importlib.util
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

spec = importlib.util.spec_from_file_location("figures_mod", Path("src/10_figures.py"))
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)

AUDIT = pd.DataFrame({"lang": ["fi", "yo", "te", "es"], "corpus_passages": [1883509, 49043, 518079, -1]})


def _bounds(langs=("fi", "yo", "te"), systems=("bm25", "mdpr", "mcontriever")):
    rows = []
    for lang in langs:
        for i, s in enumerate(systems):
            lo = 0.3 + 0.05 * i
            rows.append({"lang": lang, "tier": "x", "system": s, "lower_mean": lo, "condensed_mean": lo + 0.1,
                         "optimistic_mean": lo + 0.3})
    return pd.DataFrame(rows)


def _coverage(langs=("fi", "yo", "te"), systems=("bm25", "mdpr", "mcontriever")):
    rows = []
    for lang in langs:
        for i, s in enumerate(systems):
            rows.append({"lang": lang, "tier": "x", "system": s, "contributor": s != "mcontriever",
                         "judged10_mean": 0.5 + 0.1 * i, "ci_lo": 0.45 + 0.1 * i, "ci_hi": 0.55 + 0.1 * i})
    return pd.DataFrame(rows)


def test_language_order_follows_corpus_size_with_unknown_last():
    assert figs.order_languages(["fi", "yo", "te", "es"], AUDIT) == ["yo", "te", "fi", "es"]


def test_language_order_drops_languages_without_rows():
    assert figs.order_languages(["fi", "yo"], AUDIT) == ["yo", "fi"]


def test_main_figure_has_one_row_per_language(tmp_path):
    fig = figs.main_fig(_coverage(), _bounds(), AUDIT, None, tmp_path / "fig1")
    ax = fig.axes[0]
    assert len(ax.get_yticks()) == 3
    assert [t.get_text().split(" ")[0] for t in ax.get_yticklabels()] == ["yo", "te", "fi"]
    assert fig.get_size_inches()[0] == pytest.approx(figs.WIDTH_IN)


def test_main_figure_writes_pdf_and_png_only_where_asked(tmp_path, monkeypatch):
    figs.main_fig(_coverage(), _bounds(), AUDIT, None, tmp_path / "fig1")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fig1.pdf", "fig1.png"]
    monkeypatch.chdir(tmp_path / "..")
    before = sorted(p.name for p in (tmp_path / "..").iterdir())
    figs.main_fig(_coverage(), _bounds(), AUDIT, None, None)
    assert sorted(p.name for p in (tmp_path / "..").iterdir()) == before


def test_main_figure_draws_corrected_marker_only_when_given():
    plain = figs.main_fig(_coverage(), _bounds(), AUDIT, None, None)
    filled = pd.DataFrame({"lang": ["yo"], "system": ["bm25"], "corrected_mean": [0.5]})
    with_x = figs.main_fig(_coverage(), _bounds(), AUDIT, filled, None)
    marks = lambda fig: [ln for ln in fig.axes[1].lines if ln.get_marker() == "x"]
    assert len(marks(plain)) == 0
    assert len(marks(with_x)) == 1


def test_main_figure_skips_missing_system_cells():
    b = _bounds(langs=("fi", "yo"))
    b = b[~((b.lang == "fi") & (b.system == "mcontriever"))]
    fig = figs.main_fig(_coverage(langs=("fi", "yo")), b, AUDIT, None, None)
    assert len(fig.axes[0].get_yticks()) == 2


def test_main_figure_fills_contributors_only_in_the_coverage_panel():
    fig = figs.main_fig(_coverage(langs=("yo",)), _bounds(langs=("yo",)), AUDIT, None, None)
    fills = {ln.get_markerfacecolor() for ln in fig.axes[0].lines if ln.get_marker() == "o"}
    assert fills == {"black"}
    hollow = [ln for ln in fig.axes[0].lines if ln.get_marker() == "^"]
    assert hollow and all(ln.get_markerfacecolor() == "white" for ln in hollow)


def test_ciral_figure_has_one_panel_per_language(tmp_path):
    c = pd.DataFrame({"lang": ["hau", "hau", "swa"], "system": ["bm25dt", "mdpr", "bm25dt"],
                      "ndcg10_shallow_mean": [0.3, 0.4, 0.2], "ndcg10_deep_mean": [0.35, 0.3, 0.25]})
    fig = figs.ciral_fig(c, tmp_path / "fig3")
    assert [len(ax.get_yticks()) for ax in fig.axes[:2]] == [2, 1]
    assert (tmp_path / "fig3.pdf").exists()
