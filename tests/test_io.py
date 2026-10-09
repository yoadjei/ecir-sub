import pytest

from src.utils import io


def test_read_run_preserves_rank_order(tmp_path):
    p = tmp_path / "run.txt"
    p.write_text("q1 Q0 dB 1 9.5 tag\nq1 Q0 dA 2 8.0 tag\nq2 Q0 dC 1 1.0 tag\n", encoding="utf-8")
    run = io.read_run(p)
    assert run == {"q1": ["dB", "dA"], "q2": ["dC"]}


def test_read_run_sorts_by_rank_even_if_file_is_shuffled(tmp_path):
    p = tmp_path / "run.txt"
    p.write_text("q1 Q0 dA 2 8.0 tag\nq1 Q0 dB 1 9.5 tag\n", encoding="utf-8")
    assert io.read_run(p)["q1"] == ["dB", "dA"]


def test_qrels_roundtrip(tmp_path):
    qrels = {"q1": {"d1": 1, "d2": 0}, "q2": {"d3": 1}}
    p = tmp_path / "qrels.txt"
    io.write_qrels(qrels, p)
    assert io.read_qrels(p) == qrels


def test_run_to_scores_dict(tmp_path):
    p = tmp_path / "run.txt"
    p.write_text("q1 Q0 dB 1 9.5 tag\nq1 Q0 dA 2 8.0 tag\n", encoding="utf-8")
    assert io.read_run_scores(p) == {"q1": {"dB": 9.5, "dA": 8.0}}


def test_write_run_then_read_run_roundtrip(tmp_path):
    run = {"q1": [("dB", 9.5), ("dA", 8.25)], "q2": [("dC", 1.0)]}
    p = tmp_path / "out" / "run.txt"
    io.write_run(run, p, tag="t")
    assert io.read_run(p) == {"q1": ["dB", "dA"], "q2": ["dC"]}
    assert io.read_run_scores(p) == {"q1": {"dB": 9.5, "dA": 8.25}, "q2": {"dC": 1.0}}


def test_pyserini_qrels_types_and_size():
    pytest.importorskip("pyserini")
    qrels = io.pyserini_qrels("miracl-v1.0-yo-dev")
    assert isinstance(qrels, dict) and len(qrels) >= 100
    for qid, docs in qrels.items():
        assert isinstance(qid, str)
        assert isinstance(docs, dict)
        for docid, rel in docs.items():
            assert isinstance(docid, str)
            assert isinstance(rel, int)
