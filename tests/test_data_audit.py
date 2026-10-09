import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("audit", Path("src/01_data_audit.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_summarise_qrels_counts():
    qrels = {
        "q1": {"d1": 1, "d2": 0, "d3": 0},
        "q2": {"d4": 0},
        "q3": {"d5": 1, "d6": 1},
    }
    s = audit.summarise_qrels(qrels)
    assert s["n_queries"] == 3
    assert s["n_judgements"] == 6
    assert s["judgements_per_query_mean"] == 2.0
    assert s["judgements_per_query_median"] == 2.0
    assert s["positives_per_query_mean"] == 1.0
    assert s["queries_zero_positives"] == 1
