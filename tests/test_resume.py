import pytest

from src.utils.resume import (append_block, order_topics, read_topics,
                              remaining, truncate_to_valid, valid_prefix)

HITS = 3


def block(qid, hits=HITS, start=1):
    return [f"{qid} Q0 d{r}#0 {r} {10 - r}.500000 Faiss\n" for r in range(start, start + hits)]


def test_valid_prefix_keeps_complete_blocks_and_drops_cut_off_one():
    lines = block("1") + block("12") + block("19", hits=2)
    got = valid_prefix(lines, HITS)
    assert got.qids == ["1", "12"]
    assert got.n_lines == 6
    assert got.n_bytes == sum(len(x.encode()) for x in lines[:6])


def test_valid_prefix_drops_garbage_last_line_without_newline():
    lines = block("1") + block("12")[:2] + ["12 Q0 d3#0 3 7.5"]
    assert valid_prefix(lines, HITS).qids == ["1"]
    lines = block("1") + block("12")
    lines[-1] = lines[-1].rstrip("\n")
    assert valid_prefix(lines, HITS).qids == ["1"]


def test_valid_prefix_stops_at_wrong_rank_or_mixed_qid_or_bad_score():
    wrong_rank = block("1") + block("12", start=2)
    assert valid_prefix(wrong_rank, HITS).qids == ["1"]
    mixed = block("1") + [block("12")[0], block("13")[1], block("12")[2]]
    assert valid_prefix(mixed, HITS).qids == ["1"]
    bad_score = block("1") + [block("12")[0].replace("9.500000", "nan?"), *block("12")[1:]]
    assert valid_prefix(bad_score, HITS).qids == ["1"]


def test_valid_prefix_stops_at_repeated_qid():
    assert valid_prefix(block("1") + block("1"), HITS).qids == ["1"]


def test_valid_prefix_of_nothing_is_empty():
    got = valid_prefix([], HITS)
    assert (got.qids, got.n_lines, got.n_bytes) == ([], 0, 0)


def test_truncate_to_valid_rewrites_file_to_the_prefix(tmp_path):
    f = tmp_path / "run.txt.partial"
    f.write_text("".join(block("1") + block("12") + block("19", hits=1)), encoding="utf-8")
    assert truncate_to_valid(f, HITS) == ["1", "12"]
    assert f.read_text(encoding="utf-8") == "".join(block("1") + block("12"))


def test_truncate_to_valid_removes_a_garbage_tail_without_newline(tmp_path):
    f = tmp_path / "run.txt.partial"
    f.write_bytes("".join(block("1")).encode() + b"12 Q0 d1#0 1 9.5000\xff")
    assert truncate_to_valid(f, HITS) == ["1"]
    assert f.read_bytes() == "".join(block("1")).encode()


def test_truncate_to_valid_leaves_a_clean_file_alone(tmp_path):
    f = tmp_path / "run.txt.partial"
    text = "".join(block("1") + block("12"))
    f.write_text(text, encoding="utf-8")
    assert truncate_to_valid(f, HITS) == ["1", "12"]
    assert f.read_text(encoding="utf-8") == text


def test_remaining_keeps_topic_order():
    assert remaining(["1", "12", "19", "31"], ["12", "1"]) == ["19", "31"]
    assert remaining(["1", "12"], []) == ["1", "12"]
    assert remaining(["1", "12"], ["1", "12"]) == []


def test_append_block_writes_pyserini_format_and_round_trips(tmp_path):
    f = tmp_path / "run.txt.partial"
    hits = [("18044#0", 65.3217392), ("13378#3", 63.882324)]
    with open(f, "a", encoding="utf-8") as fh:
        append_block(fh, "1", hits, "Faiss")
        append_block(fh, "12", [("5#1", 1.0), ("6#2", 0.5)], "Faiss")
    assert f.read_text(encoding="utf-8").splitlines()[:2] == [
        "1 Q0 18044#0 1 65.321739 Faiss",
        "1 Q0 13378#3 2 63.882324 Faiss",
    ]
    assert truncate_to_valid(f, 2) == ["1", "12"]


def test_append_block_accepts_numpy_floats(tmp_path):
    np = pytest.importorskip("numpy")
    f = tmp_path / "run.txt.partial"
    with open(f, "a", encoding="utf-8") as fh:
        append_block(fh, "1", [("a", np.float32(0.25))], "Faiss")
    assert f.read_text(encoding="utf-8") == "1 Q0 a 1 0.250000 Faiss\n"


def test_read_topics_orders_numeric_ids_numerically(tmp_path):
    f = tmp_path / "topics.tsv"
    f.write_text("12\tMilloin Kiina?\n1\tMitä on altruismi?\n100\tx y\n", encoding="utf-8")
    assert read_topics(f) == [("1", "Mitä on altruismi?"), ("12", "Milloin Kiina?"), ("100", "x y")]


def test_order_topics_falls_back_to_string_order_for_non_numeric_ids():
    assert order_topics(["b", "a", "10"]) == ["10", "a", "b"]
