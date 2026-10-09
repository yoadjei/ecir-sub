import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.utils.io import write_qrels

spec = importlib.util.spec_from_file_location("llm_validate_mod", Path("src/06_llm_validate.py"))
val = importlib.util.module_from_spec(spec)
spec.loader.exec_module(val)


def big_qrels(n_queries=100):
    """2 relevant (one graded 2) and 3 non-relevant judged passages per query."""
    return {f"q{i}": {f"d{i}_{j}": (2 if j == 1 else 1) if j < 2 else 0 for j in range(5)}
            for i in range(n_queries)}


class FakeJudge:
    """returns scripted labels keyed by (qid, docid) and records the texts it was given."""

    model = "fake"

    def __init__(self, labels, seconds=2.0):
        self.labels = labels
        self.seconds = seconds
        self.seen = []

    def judge(self, qid, docid, query, passage):
        self.seen.append((qid, docid, query, passage))
        return SimpleNamespace(label=self.labels[(qid, docid)], seconds=self.seconds, cached=False)


def test_split_sizes():
    df = val.make_split(big_qrels(), "sw")
    assert len(df) == 150
    assert df.groupby(["split", "rel"]).size().to_dict() == {
        ("calibration", 0): 25, ("calibration", 1): 25, ("test", 0): 50, ("test", 1): 50}


def test_split_labels_follow_qrels():
    qrels = big_qrels()
    df = val.make_split(qrels, "sw")
    for r in df.itertuples():
        assert (qrels[r.qid][r.docid] > 0) == (r.rel == 1)


def test_split_is_deterministic():
    a = val.make_split(big_qrels(), "sw")
    b = val.make_split(big_qrels(), "sw")
    pd.testing.assert_frame_equal(a, b)


def test_split_does_not_depend_on_qrels_order():
    qrels = big_qrels()
    shuffled = dict(reversed(list(qrels.items())))
    pd.testing.assert_frame_equal(val.make_split(qrels, "sw"), val.make_split(shuffled, "sw"))


def test_split_uses_the_configured_seed():
    assert val.cfg()["seed"] == 13


def test_judged_pair_is_unique_per_row():
    df = val.make_split(big_qrels(), "sw")
    assert not df.duplicated(["qid", "docid"]).any()


def test_calibration_and_test_share_no_pair():
    df = val.make_split(big_qrels(), "sw")
    cal = set(zip(df[df.split == "calibration"].qid, df[df.split == "calibration"].docid))
    test = set(zip(df[df.split == "test"].qid, df[df.split == "test"].docid))
    assert cal and test and not cal & test


def test_short_language_uses_available_count_balanced():
    qrels = {f"q{i}": {f"p{i}": 1, f"n{i}": 0} for i in range(9)}
    qrels["q9"] = {"n9": 0, "n10": 0, "n11": 0}
    df = val.make_split(qrels, "yo")
    assert df.groupby(["split", "rel"]).size().to_dict() == {
        ("calibration", 0): 3, ("calibration", 1): 3, ("test", 0): 6, ("test", 1): 6}
    assert not df.duplicated(["qid", "docid"]).any()


def test_split_note_states_a_short_split():
    qrels = {f"q{i}": {f"p{i}": 1, f"n{i}": 0} for i in range(9)}
    note = val.split_note("yo", val.make_split(qrels, "yo"))
    assert "yo" in note and "9" in note and "75" in note
    assert val.split_note("sw", val.make_split(big_qrels(), "sw")) == ""


def test_split_without_enough_pairs_raises():
    with pytest.raises(ValueError):
        val.make_split({"q1": {"p1": 1, "n1": 0}}, "yo")


def test_load_split_writes_csv_once_and_rereads_it(tmp_path, monkeypatch):
    monkeypatch.setattr(val, "path", lambda key: tmp_path / key)
    qrels_file = tmp_path / "qrels" / "miracl.sw.dev.txt"
    write_qrels(big_qrels(), qrels_file)
    first = val.load_split("sw")
    assert (tmp_path / "results" / "llm_splits" / "sw.csv").exists()
    qrels_file.unlink()
    pd.testing.assert_frame_equal(val.load_split("sw"), first)
    assert first.rel.dtype.kind == "i"


def scripted_frame():
    rows = [("q1", "a", 1), ("q1", "b", 1), ("q2", "c", 1), ("q2", "d", 1),
            ("q3", "e", 0), ("q3", "f", 0), ("q4", "g", 0), ("q4", "h", 0)]
    return pd.DataFrame(rows, columns=["qid", "docid", "rel"])


def test_score_kappa_confusion_and_unparseable_rate():
    labels = {("q1", "a"): 1, ("q1", "b"): 1, ("q2", "c"): 0, ("q2", "d"): None,
              ("q3", "e"): 0, ("q3", "f"): 0, ("q4", "g"): 1, ("q4", "h"): 0}
    out = val.score(FakeJudge(labels), "sw", scripted_frame(), lambda q: f"query {q}", lambda d: f"text {d}")
    assert (out["tp"], out["fn"], out["tn"], out["fp"]) == (2, 2, 3, 1)
    assert out["kappa"] == pytest.approx(0.25)
    assert out["unparseable_rate"] == pytest.approx(1 / 8)
    assert out["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["recall"] == pytest.approx(0.5)
    assert out["n"] == 8 and out["lang"] == "sw" and out["model"] == "fake"
    assert out["sec_per_judgement"] == pytest.approx(2.0)


def test_score_passes_query_and_passage_text_to_the_judge():
    labels = {(q, d): r for q, d, r in scripted_frame().itertuples(index=False)}
    judge = FakeJudge(labels)
    val.score(judge, "sw", scripted_frame(), lambda q: f"query {q}", lambda d: f"text {d}")
    assert judge.seen[0] == ("q1", "a", "query q1", "text a")
    assert len(judge.seen) == 8


def test_score_perfect_judge():
    labels = {(q, d): r for q, d, r in scripted_frame().itertuples(index=False)}
    out = val.score(FakeJudge(labels), "sw", scripted_frame(), str, str)
    assert out["kappa"] == pytest.approx(1.0)
    assert out["unparseable_rate"] == 0


def calibration(kappas):
    return pd.DataFrame([{"model": m, "kappa": k} for m, ks in kappas.items() for k in ks])


def test_choose_model_takes_highest_mean_kappa():
    cal = calibration({"a": [0.1, 0.3], "b": [0.5, 0.3]})
    assert val.choose_model(cal, ["a", "b"]) == "b"


def test_choose_model_breaks_ties_by_candidate_order():
    cal = calibration({"a": [0.2, 0.4], "b": [0.4, 0.2]})
    assert val.choose_model(cal, ["a", "b"]) == "a"
    assert val.choose_model(cal, ["b", "a"]) == "b"
