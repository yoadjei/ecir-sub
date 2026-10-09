"""helpers to resume a trec run file that was cut off part way.

a run file is a list of query blocks, one block per query, each block exactly `hits`
lines with ranks 1..hits. anything after the last complete block is dropped.
"""
import math
import os
from pathlib import Path
from typing import NamedTuple


class Prefix(NamedTuple):
    qids: list[str]
    n_lines: int
    n_bytes: int


def _parse_line(line: str) -> tuple[str, int] | None:
    """(qid, rank) of a complete trec line, None for anything else."""
    if not line.endswith("\n"):
        return None
    parts = line.split()
    if len(parts) != 6 or parts[1] != "Q0":
        return None
    try:
        rank = int(parts[3])
        score = float(parts[4])
    except ValueError:
        return None
    return (parts[0], rank) if math.isfinite(score) else None


def _block_qid(block: list[str]) -> str | None:
    """qid of a complete block, None if any line is broken or the ranks are off."""
    parsed = [_parse_line(line) for line in block]
    if None in parsed:
        return None
    qids = {qid for qid, _ in parsed}
    ranks = [rank for _, rank in parsed]
    if len(qids) != 1 or ranks != list(range(1, len(block) + 1)):
        return None
    return qids.pop()


def valid_prefix(lines: list[str], hits: int) -> Prefix:
    """leading complete blocks of `hits` lines, with the line and byte count they cover."""
    qids: list[str] = []
    for start in range(0, len(lines) - hits + 1, hits):
        qid = _block_qid(lines[start:start + hits])
        if qid is None or qid in qids:
            break
        qids.append(qid)
    kept = lines[:len(qids) * hits]
    return Prefix(qids, len(kept), sum(len(line.encode("utf-8")) for line in kept))


def truncate_to_valid(path: Path, hits: int) -> list[str]:
    """cut the file in place down to its valid prefix, return the finished qids."""
    with open(path, "rb") as fh:
        raw = fh.readlines()
    lines = [line.decode("utf-8", errors="replace") for line in raw]
    prefix = valid_prefix(lines, hits)
    keep = sum(len(line) for line in raw[:prefix.n_lines])
    if keep < path.stat().st_size:
        with open(path, "r+b") as fh:
            fh.truncate(keep)
            fh.flush()
            os.fsync(fh.fileno())
    return prefix.qids


def remaining(all_qids: list[str], done_qids: list[str]) -> list[str]:
    """qids not done yet, in the original topic order."""
    done = set(done_qids)
    return [qid for qid in all_qids if qid not in done]


def append_block(fh, qid: str, hits: list[tuple[str, float]], tag: str) -> None:
    """write one query block in pyserini's trec format and force it to disk."""
    for rank, (docid, score) in enumerate(hits, start=1):
        fh.write(f"{qid} Q0 {docid} {rank} {float(score):.6f} {tag}\n")
    fh.flush()
    os.fsync(fh.fileno())


def order_topics(qids: list[str]) -> list[str]:
    """pyserini's topic order: numeric ids by value, otherwise by string."""
    if all(qid.isdigit() for qid in qids):
        return sorted(qids, key=int)
    return sorted(qids)


def read_topics(path: Path) -> list[tuple[str, str]]:
    """(qid, query) pairs from a pyserini topics tsv, in pyserini's order."""
    queries = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            qid, query = line.rstrip("\r\n").split("\t", 1)
            queries[qid] = query.strip()
    return [(qid, queries[qid]) for qid in order_topics(list(queries))]
