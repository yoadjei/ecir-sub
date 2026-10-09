import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.utils.io import read_qrels, write_qrels, write_run

spec = importlib.util.spec_from_file_location("llm_fill_mod", Path("src/07_llm_fill.py"))
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)


class FakeJudge:
    """scripted labels per (qid, docid), counts and records every call."""

    model = "fake"
    prompt_hash = "abc123"

    def __init__(self, labels, fail_after=None):
        self.labels = labels
        self.fail_after = fail_after
        self.calls = []

    def judge(self, qid, docid, query, passage):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise KeyboardInterrupt
        self.calls.append((qid, docid, query, passage))
        return SimpleNamespace(label=self.labels[(qid, docid)], seconds=1.0, cached=False)


def small_qrels():
    return {"q1": {"a": 1, "b": 0}, "q2": {"c": 1}, "q3": {"e": 0}, "q4": {"g": 1, "h": 1}}


def small_runs():
    return {"bm25": {"q1": ["x", "a", "y"], "q2": ["c", "z"], "q3": ["e", "w"], "q4": ["g", "u", "h"]},
            "mdpr": {"q1": ["a", "x", "b"], "q2": ["z", "c"], "q3": ["w"], "q4": ["v", "g"]}}


def test_sample_is_seeded_and_deterministic():
    qrels = {f"q{i}": {"d": 1} for i in range(50)}
    a = fill.sample_queries(qrels, 10, 13)
    assert a == fill.sample_queries(qrels, 10, 13)
    assert len(a) == 10 and a == sorted(a)
    assert a != fill.sample_queries(qrels, 10, 14)


def test_smaller_sample_is_inside_the_bigger_one():
    qrels = {f"q{i}": {"d": 1} for i in range(50)}
    assert set(fill.sample_queries(qrels, 10, 13)) <= set(fill.sample_queries(qrels, 25, 13))


def test_sample_ignores_qrels_order():
    qrels = {f"q{i}": {"d": 1} for i in range(50)}
    shuffled = dict(reversed(list(qrels.items())))
    assert fill.sample_queries(qrels, 10, 13) == fill.sample_queries(shuffled, 10, 13)


def test_sample_only_takes_queries_with_a_positive():
    qrels = {"q1": {"a": 0}, "q2": {"b": 1}, "q3": {"c": 0, "d": 2}}
    assert fill.sample_queries(qrels, 10, 13) == ["q2", "q3"]


def test_sample_caps_at_the_available_queries():
    qrels = {f"q{i}": {"d": 1} for i in range(5)}
    assert len(fill.sample_queries(qrels, 100, 13)) == 5


def test_holes_are_exactly_the_unjudged_top_k_docs_of_sampled_queries():
    holes = fill.find_holes(small_runs(), small_qrels(), ["q1", "q2"], 3)
    assert holes == [("q1", "x"), ("q1", "y"), ("q2", "z")]


def test_holes_respect_the_cutoff_and_are_unique():
    runs = {"bm25": {"q1": ["a", "x", "y"]}, "mdpr": {"q1": ["x", "a", "y"]}}
    assert fill.find_holes(runs, small_qrels(), ["q1"], 2) == [("q1", "x")]
    assert fill.find_holes(runs, small_qrels(), ["q1"], 3) == [("q1", "x"), ("q1", "y")]


def test_judge_holes_calls_the_judge_once_per_pair_with_texts():
    holes = [("q1", "x"), ("q2", "z")]
    judge = FakeJudge({("q1", "x"): 1, ("q2", "z"): 0})
    labels = {}
    n = fill.judge_holes(judge, holes, labels, lambda q: f"query {q}", lambda d: f"text {d}", lambda lab: None)
    assert n == 2
    assert labels == {("q1", "x"): 1, ("q2", "z"): 0}
    assert judge.calls[0] == ("q1", "x", "query q1", "text x")


def test_judge_holes_skips_pairs_that_already_have_a_label():
    judge = FakeJudge({("q2", "z"): 1})
    labels = {("q1", "x"): 1}
    n = fill.judge_holes(judge, [("q1", "x"), ("q2", "z")], labels, str, str, lambda lab: None)
    assert n == 1 and [c[:2] for c in judge.calls] == [("q2", "z")]


def test_judge_holes_checkpoints_every_n_pairs_and_at_the_end():
    holes = [("q", f"d{i}") for i in range(5)]
    judge = FakeJudge({h: 1 for h in holes})
    sizes = []
    fill.judge_holes(judge, holes, {}, str, str, lambda lab: sizes.append(len(lab)), every=2)
    assert sizes == [2, 4, 5]


def test_judge_holes_saves_progress_when_interrupted():
    holes = [("q", f"d{i}") for i in range(5)]
    judge = FakeJudge({h: 1 for h in holes}, fail_after=3)
    saved = []
    with pytest.raises(KeyboardInterrupt):
        fill.judge_holes(judge, holes, {}, str, str, lambda lab: saved.append(dict(lab)), every=50)
    assert len(saved[-1]) == 3


def test_labels_roundtrip_keeps_unparseable_as_none(tmp_path):
    file = tmp_path / "labels" / "sw.csv"
    labels = {("q1", "x"): 1, ("q1", "y"): None, ("q2", "z"): 0}
    fill.save_labels(labels, file, "fake", "abc123")
    assert fill.load_labels(file, "fake", "abc123") == labels


def test_labels_from_another_model_or_prompt_are_ignored(tmp_path):
    file = tmp_path / "sw.csv"
    fill.save_labels({("q1", "x"): 1}, file, "fake", "abc123")
    assert fill.load_labels(file, "other", "abc123") == {}
    assert fill.load_labels(file, "fake", "zzz") == {}
    assert fill.load_labels(tmp_path / "missing.csv", "fake", "abc123") == {}


def test_filled_qrels_hold_original_plus_llm_labels(tmp_path):
    qrels = small_qrels()
    holes = [("q1", "x"), ("q1", "y"), ("q2", "z")]
    labels = {("q1", "x"): 1, ("q1", "y"): None, ("q2", "z"): 0}
    filled = fill.fill_qrels(qrels, holes, labels)
    assert filled["q1"] == {"a": 1, "b": 0, "x": 1, "y": 0}
    assert filled["q2"] == {"c": 1, "z": 0}
    assert filled["q3"] == {"e": 0} and filled["q4"] == {"g": 1, "h": 1}
    assert qrels["q1"] == {"a": 1, "b": 0}
    out = tmp_path / "filled.txt"
    write_qrels(filled, out)
    assert read_qrels(out) == filled


def test_unparseable_rate_counts_none_labels_over_the_holes():
    holes = [("q", "a"), ("q", "b"), ("q", "c"), ("q", "d")]
    labels = {("q", "a"): 1, ("q", "b"): None, ("q", "c"): 0, ("q", "d"): None, ("q", "old"): None}
    assert fill.unparseable_rate(holes, labels) == 0.5
    assert fill.unparseable_rate([], labels) == 0.0


def test_rates_come_from_the_test_split_confusion_counts():
    tpr, tnr = fill.rates_from_counts({"tp": 40, "fn": 10, "tn": 45, "fp": 5})
    assert (tpr, tnr) == (0.8, 0.9)


def test_rates_without_both_classes_raise():
    with pytest.raises(ValueError):
        fill.rates_from_counts({"tp": 0, "fn": 0, "tn": 10, "fp": 0})


def test_positive_share_ignores_unparseable_labels():
    holes = [("q", "a"), ("q", "b"), ("q", "c"), ("q", "d")]
    labels = {("q", "a"): 1, ("q", "b"): None, ("q", "c"): 0, ("q", "d"): 1, ("q", "old"): 1}
    assert fill.positive_share(holes, labels) == pytest.approx(2 / 3)
    assert math.isnan(fill.positive_share(holes, {h: None for h in holes}))
    assert math.isnan(fill.positive_share([], labels))


def test_noise_model_matches_a_hand_computed_example():
    m = fill.noise_model(0.8, 0.9, 0.3)
    assert m["pi_hat"] == pytest.approx(2 / 7)
    assert m["p1"] == pytest.approx(1.6 / 2.1)
    assert m["p0"] == pytest.approx(0.4 / 4.9)
    assert m["p_unparseable"] == pytest.approx(2 / 7)
    assert m["unreliable"] is False


def test_noise_model_clips_the_prevalence_at_zero_and_one():
    low = fill.noise_model(0.8, 0.9, 0.05)
    assert low["pi_hat"] == 0.0 and low["p1"] == 0.0 and low["p0"] == 0.0
    high = fill.noise_model(0.8, 0.9, 0.95)
    assert high["pi_hat"] == 1.0 and high["p1"] == 1.0 and high["p0"] == 1.0


def test_noise_model_with_a_weak_judge_does_not_correct():
    for tpr, tnr in [(0.55, 0.45), (0.5, 0.54), (0.4, 0.5)]:
        m = fill.noise_model(tpr, tnr, 0.3)
        assert m["unreliable"] is True
        assert (m["p1"], m["p0"], m["p_unparseable"]) == (1.0, 0.0, 0.0)
        assert math.isnan(m["pi_hat"])


def test_noise_model_without_any_parseable_hole_does_not_correct():
    assert fill.noise_model(0.8, 0.9, float("nan"))["unreliable"] is True


def test_noise_model_at_balanced_prevalence_gives_ppv_and_npv():
    tpr, tnr = 0.8, 0.9
    q = 0.5 * tpr + 0.5 * (1 - tnr)
    m = fill.noise_model(tpr, tnr, q)
    assert m["pi_hat"] == pytest.approx(0.5)
    ppv = tpr / (tpr + 1 - tnr)
    npv = tnr / (tnr + 1 - tpr)
    assert m["p1"] == pytest.approx(ppv)
    assert m["p0"] == pytest.approx(1 - npv)


def test_noise_model_guards_a_zero_denominator():
    m = fill.noise_model(0.8, 1.0, 0.0)
    assert m["pi_hat"] == 0.0 and m["p1"] == 0.0 and m["p0"] == 0.0


def test_corrected_ndcg_matches_a_hand_computed_example():
    qrels = {"q1": {"a": 1, "b": 0}}
    runs = {"bm25": {"q1": ["x", "a", "y"]}}
    labels = {("q1", "x"): 1, ("q1", "y"): 0}
    model = fill.noise_model(0.8, 0.9, 0.3)
    filled = fill.fill_qrels(qrels, list(labels), labels)
    row = fill.query_rows("sw", runs, qrels, filled, labels, ["q1"], model, 3)[0]
    px, py = 1.6 / 2.1, 0.4 / 4.9
    expected = (px + 1 / math.log2(3) + py / 2) / (1 + px / math.log2(3) + py / 2)
    assert row["corrected"] == pytest.approx(expected)
    assert row["corrected"] == pytest.approx(0.9423, abs=1e-4)
    assert row["lower"] == pytest.approx(1 / math.log2(3))
    assert row["filled"] == pytest.approx(1.0)
    assert row["optimistic"] == pytest.approx(1.0)
    assert row["n_holes_top10"] == 2 and row["n_holes_llm_rel"] == 1
    assert row["has_positive"]


def test_ideal_ranking_is_shared_by_all_systems_of_a_query():
    qrels = {"q1": {"a": 1, "n": 0}}
    runs = {"bm25": {"q1": ["x", "a", "n"]}, "mdpr": {"q1": ["z", "a", "n"]}}
    labels = {("q1", "x"): 1, ("q1", "z"): 1}
    model = fill.noise_model(0.8, 0.9, 0.3)
    filled = fill.fill_qrels(qrels, list(labels), labels)
    rows = fill.query_rows("sw", runs, qrels, filled, labels, ["q1"], model, 3)
    p1 = 1.6 / 2.1
    expected = (p1 + 1 / math.log2(3)) / (1 + p1 / math.log2(3) + p1 / 2)
    assert [r["corrected"] for r in rows] == pytest.approx([expected, expected])


def test_unparseable_hole_gets_the_estimated_prevalence():
    qrels = {"q1": {"a": 1}}
    runs = {"bm25": {"q1": ["x", "a"]}}
    labels = {("q1", "x"): None}
    model = fill.noise_model(0.8, 0.9, 0.3)
    filled = fill.fill_qrels(qrels, list(labels), labels)
    row = fill.query_rows("sw", runs, qrels, filled, labels, ["q1"], model, 2)[0]
    expected = (2 / 7 + 1 / math.log2(3)) / (1 + 2 / 7 / math.log2(3))
    assert row["corrected"] == pytest.approx(expected)
    assert row["filled"] == pytest.approx(1 / math.log2(3))
    assert row["n_holes_llm_rel"] == 0 and row["n_holes_top10"] == 1


def test_unreliable_correction_falls_back_to_the_hard_labels():
    qrels = {"q1": {"a": 1, "b": 0}}
    runs = {"bm25": {"q1": ["x", "a", "y"]}}
    labels = {("q1", "x"): 1, ("q1", "y"): None}
    model = fill.noise_model(0.5, 0.5, 0.3)
    filled = fill.fill_qrels(qrels, list(labels), labels)
    row = fill.query_rows("sw", runs, qrels, filled, labels, ["q1"], model, 3)[0]
    assert row["corrected"] == pytest.approx(row["filled"])


def test_low_prevalence_lowers_the_corrected_ndcg_compared_with_ppv_and_npv():
    qrels = {"q1": {"a": 1, "b": 0}}
    runs = {"bm25": {"q1": ["x", "a", "y"]}}
    labels = {("q1", "x"): 1, ("q1", "y"): 0}
    filled = fill.fill_qrels(qrels, list(labels), labels)
    tpr, tnr = 0.8, 0.9
    balanced = fill.noise_model(tpr, tnr, 0.45)
    sparse = fill.noise_model(tpr, tnr, 0.15)
    assert sparse["pi_hat"] < 0.1
    assert sparse["p1"] < balanced["p1"] and sparse["p0"] < balanced["p0"]
    old = fill.query_rows("sw", runs, qrels, filled, labels, ["q1"], balanced, 3)[0]["corrected"]
    new = fill.query_rows("sw", runs, qrels, filled, labels, ["q1"], sparse, 3)[0]["corrected"]
    assert new < old


def test_query_rows_columns_cover_what_gaps_and_pairs_need():
    qrels, runs = small_qrels(), small_runs()
    sample = ["q1", "q2", "q4"]
    holes = fill.find_holes(runs, qrels, sample, 3)
    labels = {h: 1 for h in holes}
    filled = fill.fill_qrels(qrels, holes, labels)
    rows = pd.DataFrame(fill.query_rows("sw", runs, qrels, filled, labels, sample, fill.noise_model(0.8, 0.9, 0.3), 3))
    for col in ["lang", "tier", "system", "qid", "has_positive", "lower", "optimistic", "filled", "corrected",
                "n_holes_top10", "n_holes_llm_rel"]:
        assert col in rows.columns
    assert len(rows) == 2 * 3
    assert set(rows.tier) == {"low"}


def test_gate_langs_skips_languages_that_failed():
    val = pd.DataFrame({"lang": ["sw", "yo", "th"], "passes_gate": [True, False, True]})
    assert fill.gate_langs(val, None) == (["sw", "th"], ["yo"])
    assert fill.gate_langs(val, ["yo", "th"]) == (["th"], ["yo"])
    assert fill.gate_langs(val, ["fi"]) == ([], ["fi"])


def test_estimate_hours():
    assert fill.estimate_hours(3600, 1.0) == pytest.approx(1.0)
    assert fill.estimate_hours(1800, 2.0) == pytest.approx(1.0)


def ql_frame(lang, tier_name, systems, n=6):
    rows = []
    for s_i, system in enumerate(systems):
        for i in range(n):
            base = 0.2 + 0.1 * s_i + 0.03 * i
            rows.append({"lang": lang, "tier": tier_name, "system": system, "qid": f"q{i}", "has_positive": True,
                         "lower": base, "optimistic": base + 0.2, "filled": base + 0.05, "corrected": base + 0.06,
                         "n_holes_top10": 4, "n_holes_llm_rel": 1})
    return pd.DataFrame(rows)


def test_summarise_reports_means_holes_and_rates():
    ql = ql_frame("sw", "low", ["bm25"])
    stats = {"sw": {"holes": 20, "unparseable_rate": 0.05, "tpr": 0.8, "tnr": 0.9, "q": 0.3,
                    "pi_hat": 2 / 7, "correction_unreliable": False}}
    out = fill.summarise(ql, stats)
    row = out.iloc[0]
    assert row["holes_per_query"] == 4 and row["llm_rel_rate"] == 0.25
    assert row["unparseable_rate"] == 0.05 and row["unique_holes"] == 20
    assert row["lower_mean"] == pytest.approx(ql.lower.mean(), abs=1e-4)
    assert row["lower_lo"] <= row["lower_mean"] <= row["lower_hi"]
    assert row["corrected_mean"] == pytest.approx(ql.corrected.mean(), abs=1e-4)
    assert (row["tpr"], row["tnr"], row["q"]) == (0.8, 0.9, 0.3)
    assert row["pi_hat"] == pytest.approx(0.2857, abs=1e-4) and not row["correction_unreliable"]
    assert "precision" not in out.columns and "ppv" not in out.columns


def test_gaps_and_pairs_accept_the_filled_treatments():
    ql = pd.concat([ql_frame("sw", "low", ["bm25", "mdpr"]), ql_frame("fi", "high", ["bm25", "mdpr"])])
    g = fill.bounds.gaps(ql, treatments=fill.TREATMENTS)
    assert set(g.treatment) == set(fill.TREATMENTS)
    p = fill.bounds.pairs(ql, treatments=("lower", "corrected"))
    assert set(p.treatment) == {"lower", "corrected"}


def build_tree(root, langs):
    qrels = {f"q{i}": {f"p{i}": 1, f"n{i}": 0} for i in range(6)}
    for lang in langs:
        write_qrels(qrels, root / "qrels" / f"miracl.{lang}.dev.txt")
        for system, offset in (("bm25", 0), ("mdpr", 1)):
            run = {f"q{i}": [(f"p{i}", 3.0), (f"h{i}_{system}", 2.0), (f"n{i}", 1.0)] if offset == 0
                   else [(f"h{i}_{system}", 3.0), (f"p{i}", 2.0), (f"h{i}_shared", 1.0)] for i in range(6)}
            write_run(run, root / "runs" / f"miracl.{lang}.{system}.txt", system)
    return qrels


def all_labels():
    return {(f"q{i}", d): (i + len(d)) % 2 for i in range(6) for d in
            [f"h{i}_bm25", f"h{i}_mdpr", f"h{i}_shared"]}


def write_validation(root, rows):
    (root / "results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "results" / "llm_validation.csv", index=False)


def validation_rows():
    return [{"lang": "sw", "model": "fake", "kappa": 0.5, "tp": 40, "fn": 10, "tn": 45, "fp": 5,
             "sec_per_judgement": 2.0, "passes_gate": True},
            {"lang": "fi", "model": "fake", "kappa": 0.4, "tp": 25, "fn": 25, "tn": 24, "fp": 26,
             "sec_per_judgement": 3.0, "passes_gate": True},
            {"lang": "yo", "model": "fake", "kappa": 0.1, "tp": 30, "fn": 20, "tn": 30, "fp": 20,
             "sec_per_judgement": 1.0, "passes_gate": False}]


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(fill, "path", lambda key: tmp_path / key)
    build_tree(tmp_path, ["sw", "fi", "yo"])
    write_validation(tmp_path, validation_rows())
    judges = []

    def use_judge(fail_after=None):
        def make_judge(model, prompt, cache):
            judges.append(FakeJudge(all_labels(), fail_after))
            return judges[-1]
        monkeypatch.setattr(fill, "Judge", make_judge)

    use_judge()
    monkeypatch.setattr(fill, "query_text", lambda lang, q: f"query {q}")
    monkeypatch.setattr(fill, "passage_text", lambda lang, d: f"text {d}")
    return SimpleNamespace(root=tmp_path, judges=judges, use_judge=use_judge)


def test_main_writes_every_output_and_skips_the_failed_language(tree, capsys):
    fill.main(["--sample-queries", "4"])
    res = tree.root / "results"
    for name in ["filled_query_level.csv", "filled.csv", "gaps_filled.csv", "pairs_filled.csv"]:
        assert (res / name).exists()
    assert (tree.root / "qrels" / "miracl.sw.filled.txt").exists()
    assert (tree.root / "qrels" / "miracl.fi.filled.txt").exists()
    assert not (tree.root / "qrels" / "miracl.yo.filled.txt").exists()
    summary = pd.read_csv(res / "filled.csv")
    assert set(summary.lang) == {"sw", "fi"}
    assert set(summary.system) == {"bm25", "mdpr"}
    assert "skipped" in capsys.readouterr().out


def test_main_reports_rates_prevalence_and_flags_a_weak_judge(tree, capsys):
    fill.main(["--sample-queries", "4"])
    out = capsys.readouterr().out
    summary = pd.read_csv(tree.root / "results" / "filled.csv")
    sw, fi = summary[summary.lang == "sw"].iloc[0], summary[summary.lang == "fi"].iloc[0]
    assert (sw.tpr, sw.tnr) == (0.8, 0.9)
    ql = pd.read_csv(tree.root / "results" / "filled_query_level.csv")
    sampled = sorted(set(ql[ql.lang == "sw"].qid))
    q = sum(int(qid[1:]) % 2 == 0 for qid in sampled) / len(sampled)
    assert sw.q == q and sw.pi_hat == pytest.approx(min(1, max(0, (q - 0.1) / 0.7)), abs=1e-4)
    assert not sw.correction_unreliable and fi.correction_unreliable
    assert (fi.tpr, fi.tnr) == (0.5, 0.48) and math.isnan(fi.pi_hat)
    assert "correction unreliable" in out
    weak = ql[ql.lang == "fi"]
    assert weak.corrected.tolist() == pytest.approx(weak.filled.tolist())
    sw_rows = ql[ql.lang == "sw"]
    assert not (sw_rows.corrected - sw_rows.filled).abs().lt(1e-9).all()


def test_main_samples_only_the_requested_number_of_queries(tree):
    fill.main(["--sample-queries", "4"])
    ql = pd.read_csv(tree.root / "results" / "filled_query_level.csv", dtype={"qid": str})
    assert ql.groupby(["lang", "system"]).qid.nunique().eq(4).all()
    filled = read_qrels(tree.root / "qrels" / "miracl.sw.filled.txt")
    sampled = set(ql[ql.lang == "sw"].qid)
    for qid, docs in filled.items():
        has_llm_labels = len(docs) > 2
        assert has_llm_labels == (qid in sampled)


def test_second_run_makes_zero_judge_calls(tree):
    fill.main(["--sample-queries", "4"])
    first = tree.judges[0]
    assert len(first.calls) == 2 * 12
    fill.main(["--sample-queries", "4"])
    assert len(tree.judges[1].calls) == 0


def test_interrupted_run_resumes_with_only_the_missing_pairs(tree):
    tree.use_judge(fail_after=5)
    with pytest.raises(KeyboardInterrupt):
        fill.main(["--langs", "sw", "--sample-queries", "4"])
    done = len(tree.judges[0].calls)
    assert done == 5
    tree.use_judge()
    fill.main(["--langs", "sw", "--sample-queries", "4"])
    resumed = {c[:2] for c in tree.judges[1].calls}
    already = {c[:2] for c in tree.judges[0].calls}
    assert not resumed & already
    assert len(resumed) + done == 4 * 3


def test_main_labels_are_the_fake_judge_labels_in_the_filled_qrels(tree):
    fill.main(["--langs", "sw", "--sample-queries", "6"])
    filled = read_qrels(tree.root / "qrels" / "miracl.sw.filled.txt")
    labels = all_labels()
    for (qid, docid), lab in labels.items():
        assert filled[qid][docid] == lab


def test_dry_run_counts_holes_and_never_judges(tree, capsys):
    fill.main(["--dry-run", "--sample-queries", "4"])
    out = capsys.readouterr().out
    assert "sw" in out and "fi" in out and "12" in out
    assert "0.02 h" in out
    assert tree.judges[0].calls == []
    assert not (tree.root / "results" / "filled.csv").exists()
    assert not (tree.root / "qrels" / "miracl.sw.filled.txt").exists()
