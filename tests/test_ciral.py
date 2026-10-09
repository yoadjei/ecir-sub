import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("ciral_mod", Path("src/08_ciral.py"))
ciral = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ciral)


@pytest.fixture(autouse=True)
def small_counts(monkeypatch):
    monkeypatch.setitem(ciral.cfg(), "bootstrap_resamples", 300)
    monkeypatch.setitem(ciral.cfg(), "permutations", 300)


def test_rrf_fuse_orders_by_summed_reciprocal_rank():
    a = {"q1": ["d1", "d2", "d3"]}
    b = {"q1": ["d3", "d2", "d4"]}
    fused = ciral.rrf_fuse([a, b], rrf_k=60, depth=10)
    assert fused["q1"] == ["d3", "d2", "d1", "d4"]


def test_rrf_fuse_ties_break_by_docid_and_depth_cuts():
    a = {"q1": ["d2", "d1"]}
    b = {"q1": ["d1", "d2"]}
    assert ciral.rrf_fuse([a, b], depth=10)["q1"] == ["d1", "d2"]
    assert ciral.rrf_fuse([a, b], depth=1)["q1"] == ["d1", "d2"]
    assert ciral.rrf_fuse([a], depth=1)["q1"] == ["d2"]


def test_rrf_fuse_keeps_queries_missing_from_one_run():
    a = {"q1": ["d1"], "q2": ["d9"]}
    b = {"q1": ["d1"]}
    fused = ciral.rrf_fuse([a, b])
    assert fused["q2"] == ["d9"]


def test_query_level_two_qrels():
    run = {"q1": ["a", "b", "c"], "q2": ["x", "y", "z"]}
    shallow = {"q1": {"a": 1, "b": 0}, "q2": {"x": 1}}
    deep = {"q1": {"a": 1, "b": 0, "c": 1}, "q2": {"x": 0, "y": 1}}
    ql = ciral.query_level("ha", "sysA", run, shallow, deep).set_index("qid")
    assert ql.loc["q1", "judged10_shallow"] == pytest.approx(2 / 10)
    assert ql.loc["q1", "judged10_deep"] == pytest.approx(3 / 10)
    assert ql.loc["q1", "ndcg10_shallow"] == pytest.approx(1.0)
    expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert ql.loc["q1", "ndcg10_deep"] == pytest.approx(expected)
    assert ql.loc["q2", "ndcg20_deep"] == pytest.approx(1 / np.log2(3))
    assert list(ql.columns) == ["lang", "system", "judged10_shallow", "judged10_deep",
                                "ndcg10_shallow", "ndcg10_deep", "ndcg20_shallow", "ndcg20_deep"]


def test_query_level_query_without_results_scores_zero():
    shallow = {"q1": {"a": 1}}
    deep = {"q1": {"a": 1}}
    ql = ciral.query_level("ha", "sysA", {}, shallow, deep)
    assert ql.iloc[0]["ndcg20_shallow"] == 0.0
    assert ql.iloc[0]["judged10_deep"] == 0.0


def test_query_level_uses_only_queries_in_both_qrels():
    run = {"q1": ["a"], "q2": ["a"]}
    ql = ciral.query_level("ha", "sysA", run, {"q1": {"a": 1}}, {"q1": {"a": 1}, "q2": {"a": 1}})
    assert list(ql.qid) == ["q1"]


def test_summarise_columns_and_counts():
    ql = pd.DataFrame({"lang": "ha", "system": "sysA", "qid": [f"q{i}" for i in range(20)],
                       "judged10_shallow": 0.2, "judged10_deep": 0.6,
                       "ndcg10_shallow": 0.3, "ndcg10_deep": 0.1,
                       "ndcg20_shallow": 0.4, "ndcg20_deep": 0.2})
    out = ciral.summarise(ql)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_queries"] == 20
    assert row["judged10_shallow_mean"] == pytest.approx(0.2)
    assert row["judged10_deep_lo"] == pytest.approx(0.6)
    assert row["ndcg20_deep_hi"] == pytest.approx(0.2)


def test_rank_systems_orders_by_mean():
    summ = pd.DataFrame({"lang": ["ha"] * 3, "system": ["a", "b", "c"],
                         "ndcg10_shallow_mean": [0.2, 0.3, 0.1], "ndcg10_deep_mean": [0.5, 0.1, 0.3]})
    ranks = ciral.rank_systems(summ, "ndcg10_shallow_mean")
    assert ranks.to_dict() == {"b": 1, "a": 2, "c": 3}
    assert ciral.rank_systems(summ, "ndcg10_deep_mean").to_dict() == {"a": 1, "c": 2, "b": 3}


def _ql(lang, system, shallow, deep, n=40, seed=0):
    """query level frame with ndcg10 and ndcg20 set to given means plus noise."""
    noise = np.random.default_rng(seed).uniform(-0.02, 0.02, size=n)
    return pd.DataFrame({"lang": lang, "system": system, "qid": [f"q{i}" for i in range(n)],
                         "judged10_shallow": 0.2, "judged10_deep": 0.5,
                         "ndcg10_shallow": shallow + noise, "ndcg10_deep": deep + noise,
                         "ndcg20_shallow": shallow + noise, "ndcg20_deep": deep + noise})


def test_pair_table_detects_sign_flip_and_significance_change():
    ql = pd.concat([_ql("ha", "a", 0.30, 0.20, seed=1), _ql("ha", "b", 0.20, 0.30, seed=2),
                    _ql("ha", "c", 0.10, 0.10, seed=3)])
    pairs = ciral.pair_table(ql, ["a", "b", "c"], metric="ndcg10")
    ab = pairs[(pairs.sys_a == "a") & (pairs.sys_b == "b")].iloc[0]
    assert ab["mean_diff_shallow"] == pytest.approx(0.10, abs=0.02)
    assert ab["mean_diff_deep"] == pytest.approx(-0.10, abs=0.02)
    assert ab["sign_flip"]
    assert ab["sig_shallow"] and ab["sig_deep"]
    assert not ab["sig_change"]
    assert ab["flip_one_sig"] and ab["flip_both_sig"]
    ac = pairs[(pairs.sys_a == "a") & (pairs.sys_b == "c")].iloc[0]
    assert ac["sig_shallow"] and ac["sig_deep"]
    assert not ac["sign_flip"]
    assert not ac["flip_one_sig"] and not ac["flip_both_sig"]
    assert len(pairs) == 3
    assert set(pairs.metric) == {"ndcg10"}


def test_pair_table_significance_change_without_sign_flip():
    a = _ql("ha", "a", 0.30, 0.20, seed=1)
    b = _ql("ha", "b", 0.20, 0.20, seed=2)
    b["ndcg20_deep"] = a["ndcg20_deep"].values
    row = ciral.pair_table(pd.concat([a, b]), ["a", "b"], metric="ndcg20").iloc[0]
    assert row["sig_shallow"] and not row["sig_deep"]
    assert row["sig_change"]
    assert not row["sign_flip"]
    assert not row["flip_one_sig"]


def test_pair_table_holm_is_applied_within_language():
    ql = pd.concat([_ql("ha", "a", 0.30, 0.30, seed=1), _ql("ha", "b", 0.30, 0.30, seed=2),
                    _ql("ha", "c", 0.30, 0.30, seed=3)])
    pairs = ciral.pair_table(ql, ["a", "b", "c"], metric="ndcg10")
    assert (pairs.p_holm_shallow >= pairs.p_raw_shallow - 1e-12).all()


def test_qrels_stats_overlap_and_agreement():
    shallow = {"q1": {"a": 1, "b": 0, "z": 1}}
    deep = {"q1": {"a": 1, "b": 1, "c": 0}}
    s = ciral.qrels_stats(shallow, deep)
    assert s["queries"] == 1
    assert s["shallow_judgements"] == 3 and s["deep_judgements"] == 3
    assert s["shallow_positives"] == 2 and s["deep_positives"] == 2
    assert s["shallow_in_deep"] == 2
    assert s["label_agree"] == 1


def test_pool_share_counts_judged_pairs_in_top_k():
    run = {"q1": ["a", "b", "c"], "q2": ["x"]}
    qrels = {"q1": {"a": 1, "c": 0}, "q2": {"y": 1}}
    assert ciral.pool_share(run, qrels, 2) == pytest.approx(1 / 3)
    assert ciral.pool_share(run, qrels, 3) == pytest.approx(2 / 3)
    assert ciral.pool_share({}, {}, 3) == 0.0


def test_qrels_counts():
    q = {"q1": {"a": 1, "b": 0}, "q2": {"c": 1}}
    assert ciral.qrels_counts(q) == {"queries": 2, "judgements": 3, "positives": 2}


def test_cli_command_uses_2cr_names():
    cmd = ciral.cli_command("yo", "bm25qt", "b")
    assert "ciral-v1.0-yo-test-b-native" in cmd
    assert cmd[cmd.index("--index") + 1] == "ciral-v1.0-yo"
    cmd = ciral.cli_command("sw", "bm25dt", "a")
    assert "ciral-v1.0-sw-test-a" in cmd
    assert cmd[cmd.index("--index") + 1] == "ciral-v1.0-sw-en"
    cmd = ciral.cli_command("so", "mdpr", "a")
    assert cmd[cmd.index("--index") + 1] == "ciral-v1.0-so-mdpr-tied-pft-msmarco"
