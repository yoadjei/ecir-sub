import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("e5", Path("src/03_runs_e5.py"))
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)


def test_passage_text_has_prefix_title_and_text():
    assert e5.passage_text("Kamaru", "a fighter") == "passage: Kamaru a fighter"


def test_passage_text_without_title_has_no_trailing_space():
    assert e5.passage_text("", "a fighter") == "passage: a fighter"
    assert e5.passage_text("Kamaru", "") == "passage: Kamaru"


def test_query_text_has_prefix():
    assert e5.query_text("Omo orile ede wo?") == "query: Omo orile ede wo?"


def test_build_run_maps_rows_back_to_docids():
    docids = ["d0", "d1", "d2", "d3"]
    scores = np.array([[0.9, 0.5], [0.8, 0.1]], dtype=np.float32)
    idx = np.array([[2, 0], [3, 1]])
    run = e5.build_run(["q1", "q2"], scores, idx, docids)
    assert [d for d, _ in run["q1"]] == ["d2", "d0"]
    assert [d for d, _ in run["q2"]] == ["d3", "d1"]
    assert run["q1"][0][1] == np.float32(0.9)
    assert isinstance(run["q1"][0][1], float)


def test_build_run_keeps_query_order():
    scores = np.zeros((2, 1), dtype=np.float32)
    idx = np.zeros((2, 1), dtype=np.int64)
    run = e5.build_run(["b", "a"], scores, idx, ["d0"])
    assert list(run) == ["b", "a"]
