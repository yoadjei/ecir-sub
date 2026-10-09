"""trec run and qrels io, keys are always strings."""
from collections import defaultdict
from pathlib import Path


def read_run(path: Path) -> dict[str, list[str]]:
    """run file to {qid: [docid, ...]} ordered by the rank column."""
    tmp: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, _, _ = parts[:6]
            tmp[qid].append((int(rank), docid))
    return {qid: [d for _, d in sorted(pairs)] for qid, pairs in tmp.items()}


def read_run_scores(path: Path) -> dict[str, dict[str, float]]:
    """run file to {qid: {docid: score}}, the shape ir_measures takes."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, _, score, _ = parts[:6]
            out[qid][docid] = float(score)
    return dict(out)


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            out[qid][docid] = int(rel)
    return dict(out)


def write_qrels(qrels: dict[str, dict[str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for qid in qrels:
            for docid, rel in qrels[qid].items():
                fh.write(f"{qid} 0 {docid} {rel}\n")


def write_run(run: dict[str, list[tuple[str, float]]], path: Path, tag: str) -> None:
    """{qid: [(docid, score), ...]} already sorted by score, written as a run file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for qid, hits in run.items():
            for rank, (docid, score) in enumerate(hits, start=1):
                fh.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")


def pyserini_qrels(name: str) -> dict[str, dict[str, int]]:
    """qrels set registered in pyserini, ids as str and labels as int."""
    from pyserini.search import get_qrels
    raw = get_qrels(name)
    return {str(q): {str(d): int(r) for d, r in docs.items()} for q, docs in raw.items()}
