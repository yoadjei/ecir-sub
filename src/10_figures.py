"""figures from results/*.csv, greyscale safe (shapes and fill carry the meaning, not colour).

fig1_main   judged@10 with ci (filled marker for pool contributors) next to ndcg@10 from lower to
            optimistic, condensed tick
fig2_ciral  ndcg@10 under shallow and deep qrels per language and system (only if ciral.csv exists)

languages run from the smallest corpus (top) to the largest. every function takes an output stem
and writes stem.pdf and stem.png, or nothing when the stem is None.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from src.utils.config import ROOT, cfg

WIDTH_IN = 4.72
ROW_IN = 0.38
CIRAL_ROW_IN = 0.17
MARKERS = ["o", "s", "^", "D", "v", "P"]
SHORT = {"bm25": "BM25", "mdpr": "mDPR", "mcontriever": "mContriever", "e5small": "mE5-small"}
RC = {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
      "pdf.fonttype": 42, "axes.linewidth": 0.6}


def order_languages(present, audit: pd.DataFrame) -> list[str]:
    """present languages by corpus size, smallest first. a missing size (-1) goes last."""
    size = audit.set_index("lang").corpus_passages
    known = sorted((l for l in present if size.get(l, -1) > 0), key=lambda l: size[l])
    return known + [l for l in present if l not in known]


def systems_in(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df.system))
    known = [s for s in cfg()["systems"] if s in present]
    return known + [s for s in present if s not in known]


def row_offsets(n: int, step: float = 0.17) -> list[float]:
    return [(i - (n - 1) / 2) * step for i in range(n)]


def style_axes(ax, rows: int, xlim: tuple, xticks: list) -> None:
    """row bands, grid and spines shared by every panel."""
    ax.set_ylim(rows - 0.5, -0.5)
    for i in range(1, rows, 2):
        ax.axhspan(i - 0.5, i + 0.5, color="0.94", zorder=0, lw=0)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.grid(axis="x", color="0.85", lw=0.4, zorder=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def shape_handles(systems: list[str]) -> list[Line2D]:
    return [Line2D([], [], marker=MARKERS[i], color="black", ls="", ms=4.5, label=SHORT.get(s, s)) for i, s in enumerate(systems)]


def add_legends(fig, systems: list[str], glyphs: list[Line2D]) -> None:
    """one legend, system shapes on the top row and what fill and ticks mean below."""
    shapes = shape_handles(systems)
    blank = Line2D([], [], ls="", label=" ")
    n = max(len(shapes), len(glyphs))
    shapes += [blank] * (n - len(shapes))
    glyphs = glyphs + [blank] * (n - len(glyphs))
    handles = [h for pair in zip(shapes, glyphs) for h in pair]
    fig.legend(handles=handles, loc="outside lower center", ncol=n, frameon=False, handletextpad=0.2, columnspacing=0.9)


def save(fig, stem) -> None:
    if stem is None:
        return
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)


def lang_labels(df: pd.DataFrame, order: list[str]) -> list[str]:
    tier = df.drop_duplicates("lang").set_index("lang").tier
    return [f"{l} ({tier[l]})" for l in order]


def draw_bounds(ax, bnd: pd.DataFrame, order: list[str], systems: list[str]) -> None:
    offsets = row_offsets(len(systems))
    cells = bnd.set_index(["lang", "system"])
    for row, lang in enumerate(order):
        for i, s in enumerate(systems):
            if (lang, s) not in cells.index:
                continue
            c, y, m = cells.loc[(lang, s)], row + offsets[i], MARKERS[i]
            ax.plot([c.lower_mean, c.optimistic_mean], [y, y], color="0.45", lw=1.0, zorder=2)
            ax.plot(c.condensed_mean, y, marker="|", color="0.35", ms=6, mew=1.0, ls="", zorder=3)
            ax.plot(c.lower_mean, y, marker=m, mfc="black", mec="black", ms=4, ls="", zorder=4)
            ax.plot(c.optimistic_mean, y, marker=m, mfc="white", mec="black", ms=4, mew=0.8, ls="", zorder=4)
    ax.set_xlabel("nDCG@10")


def draw_coverage(ax, cov: pd.DataFrame, order: list[str], systems: list[str]) -> None:
    offsets = row_offsets(len(systems))
    cells = cov.set_index(["lang", "system"])
    for row, lang in enumerate(order):
        for i, s in enumerate(systems):
            if (lang, s) not in cells.index:
                continue
            c, y = cells.loc[(lang, s)], row + offsets[i]
            ax.plot([c.ci_lo, c.ci_hi], [y, y], color="0.45", lw=1.0, zorder=2)
            ax.plot(c.judged10_mean, y, marker=MARKERS[i], mfc="black" if c.contributor else "white", mec="black",
                    ms=4, mew=0.8, ls="", zorder=4)
    ax.set_xlabel("Judged@10")


def bound_glyphs() -> list[Line2D]:
    glyphs = [Line2D([], [], marker="o", mfc="black", mec="black", color="black", ls="", ms=4, label="lower"),
              Line2D([], [], marker="o", mfc="white", mec="black", color="black", ls="", ms=4, label="optimistic"),
              Line2D([], [], marker="|", color="0.35", ls="", ms=6, mew=1.0, label="condensed")]
    return glyphs


def main_fig(cov: pd.DataFrame, bnd: pd.DataFrame, audit: pd.DataFrame, stem) -> plt.Figure:
    """judged@10 and the nDCG@10 bounds side by side, one row per language."""
    order = order_languages(list(dict.fromkeys(bnd.lang)), audit)
    systems = systems_in(bnd)
    with plt.rc_context(RC):
        fig, (axc, axb) = plt.subplots(1, 2, sharey=True, width_ratios=[1, 1.7], layout="constrained",
                                       figsize=(WIDTH_IN, ROW_IN * len(order) + 0.95))
        axc.set_yticks(range(len(order)))
        axc.set_yticklabels(lang_labels(bnd, order))
        style_axes(axc, len(order), (0, 0.5), [0, 0.1, 0.2, 0.3, 0.4, 0.5])
        style_axes(axb, len(order), (0, 1.03), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        draw_coverage(axc, cov, order, systems)
        draw_bounds(axb, bnd, order, systems)
        add_legends(fig, systems, bound_glyphs())
    save(fig, stem)
    return fig


def ciral_fig(ciral: pd.DataFrame, stem) -> plt.Figure:
    """one small panel per language, a dumbbell from shallow to deep for each system."""
    langs = list(dict.fromkeys(ciral.lang))
    systems = systems_in(ciral)
    tall = max((ciral.lang == l).sum() for l in langs)
    ncols = 2
    nrows = -(-len(langs) // ncols)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False, layout="constrained",
                                 figsize=(WIDTH_IN, nrows * (CIRAL_ROW_IN * tall + 0.3) + 0.6))
        for ax in axes.flat[len(langs):]:
            ax.set_visible(False)
        for ax, lang in zip(axes.flat, langs):
            cells = ciral[ciral.lang == lang].reset_index(drop=True)
            ax.set_title(lang, fontsize=8, loc="left")
            ax.set_yticks(range(len(cells)))
            ax.set_yticklabels(cells.system)
            ax.set_ylim(tall - 0.5, -0.5)
            ax.set_xlim(0, 0.5)
            ax.grid(axis="x", color="0.85", lw=0.4, zorder=0)
            ax.tick_params(axis="y", length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for row, c in cells.iterrows():
                m = MARKERS[systems.index(c.system)]
                ax.plot([c.ndcg10_shallow_mean, c.ndcg10_deep_mean], [row, row], color="0.45", lw=1.0, zorder=2)
                ax.plot(c.ndcg10_shallow_mean, row, marker=m, mfc="white", mec="black", ms=4, mew=0.8, ls="", zorder=4)
                ax.plot(c.ndcg10_deep_mean, row, marker=m, mfc="black", mec="black", ms=4, ls="", zorder=4)
        for ax in axes[-1]:
            ax.set_xlabel("nDCG@10")
        glyphs = [Line2D([], [], marker="o", mfc="white", mec="black", color="black", ls="", ms=4, label="shallow"),
                  Line2D([], [], marker="o", mfc="black", mec="black", color="black", ls="", ms=4, label="deep")]
        fig.legend(handles=glyphs, loc="outside lower center", ncol=2, frameon=False, handletextpad=0.2)
    save(fig, stem)
    return fig


def read_optional(name: str) -> pd.DataFrame | None:
    file = ROOT / "results" / name
    return pd.read_csv(file) if file.exists() else None


def main() -> None:
    out = ROOT / "figures"
    audit = pd.read_csv(ROOT / "results" / "data_audit.csv")
    main_fig(pd.read_csv(ROOT / "results" / "coverage.csv"), pd.read_csv(ROOT / "results" / "bounds.csv"), audit,
             out / "fig1_main")
    ciral = read_optional("ciral.csv")
    if ciral is not None:
        ciral_fig(ciral, out / "fig2_ciral")
    print(f"[fig] wrote {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
