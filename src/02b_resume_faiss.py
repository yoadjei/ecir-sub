"""dense search for the miracl runs that can be stopped and restarted.

usage: PYTHONPATH=. python src/02b_resume_faiss.py --lang fi --system mcontriever [--batch-size 32]
gives the same run as `python -m pyserini.search.faiss` but loads the index once and appends
every finished query to runs/miracl.<lang>.<system>.txt.partial (fsync after each query). a restart
trims the partial file to its last complete query and carries on. when every topic is done the
partial file is renamed to the final run file.

works offline. pyserini 2.4 downloads index metadata from github as soon as it is imported, so
it is not used here: the hugging face model is run with transformers (same tokenizer call, pooling
and no normalisation as pyserini's AutoQueryEncoder) and the prebuilt index is read with faiss.
nothing is downloaded, a missing index, topics file or model stops the script with a message.
--verify N searches the first N finished queries again and compares them with the partial file.
"""
import os

# must be set before transformers is imported
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from src.utils.config import cfg, path  # noqa: E402
from src.utils.io import read_run_scores  # noqa: E402
from src.utils.resume import (append_block, read_topics, remaining,  # noqa: E402
                              truncate_to_valid)

TAG = "Faiss"
SCORE_TOLERANCE = 1e-4
# pooling the pyserini cli uses for each encoder class in config.yaml
POOLING = {"contriever": "mean", "auto": "cls"}


def cache_home() -> Path:
    return Path(os.environ.get("PYSERINI_CACHE") or Path.home() / ".cache" / "pyserini")


def index_dir(lang: str, system: str, index_name: str) -> Path:
    """unpacked prebuilt index folder, named faiss.miracl-v1.0-<lang>.<system>-...<md5>."""
    pattern = f"faiss.miracl-v1.0-{lang}.{system}-*"
    found = sorted((cache_home() / "indexes").glob(pattern))
    found = [folder for folder in found if (folder / "index").is_file() and (folder / "docid").is_file()]
    if len(found) != 1:
        sys.exit(f"missing offline: need one folder {cache_home() / 'indexes' / pattern} holding index and docid "
                 f"(found {len(found)}, prebuilt index {index_name}); download it once while online")
    return found[0]


def topics_file(topics_name: str) -> Path:
    found = cache_home() / "topics" / f"topics.{topics_name}.tsv"
    if not found.is_file():
        sys.exit(f"missing offline: {found} (topics {topics_name}); fetch them once while online")
    return found


class DenseSearcher:
    """query encoder and flat inner product index, used one batch at a time."""

    def __init__(self, system: dict, index: Path):
        if system["encoder_class"] not in POOLING:
            sys.exit(f"encoder class {system['encoder_class']} is not supported here")
        self.pooling = POOLING[system["encoder_class"]]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(system["encoder"], clean_up_tokenization_spaces=True)
            self.model = AutoModel.from_pretrained(system["encoder"])
        except OSError as err:
            sys.exit(f"missing offline: encoder {system['encoder']} is not in the hugging face cache ({err})")
        self.index = faiss.read_index(str(index / "index"))
        with open(index / "docid", encoding="utf-8") as fh:
            self.docids = [line.rstrip() for line in fh]

    def encode(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(query, add_special_tokens=True, return_tensors="pt", padding="longest",
                                return_token_type_ids=False, truncation=True)
        with torch.no_grad():
            states = self.model(**inputs)[0].numpy()
        vector = np.average(states, axis=-2) if self.pooling == "mean" else states[:, 0, :]
        return vector.flatten()

    def search(self, qids: list[str], texts: list[str], hits: int, threads: int) -> list[tuple[str, list]]:
        vectors = np.array([self.encode(text) for text in texts]).astype("float32")
        faiss.omp_set_num_threads(threads)
        scores, rows = self.index.search(vectors, hits)
        return [(qid, [(self.docids[row], score) for score, row in zip(qscores, qrows) if row != -1])
                for qid, qscores, qrows in zip(qids, scores, rows)]


def stop_on_signal() -> threading.Event:
    """set the event on sigint or sigterm, a second signal kills the process as usual."""
    stop = threading.Event()

    def handler(signum, frame):
        stop.set()
        signal.signal(signum, signal.SIG_DFL)
        print("[resume] stop requested, finishing the current batch", flush=True)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    return stop


def verify(searcher: DenseSearcher, partial: Path, queries: dict[str, str], count: int, args) -> int:
    """search the first `count` finished queries again, return how many hit lines differ."""
    old = read_run_scores(partial)
    qids = list(old)[:count]
    bad = 0
    for start in range(0, len(qids), args.batch_size):
        batch = qids[start:start + args.batch_size]
        for qid, found in searcher.search(batch, [queries[q] for q in batch], args.hits, args.threads):
            before = list(old[qid].items())
            if len(found) != len(before):
                bad += 1
                print(f"[verify] qid {qid} has {len(found)} new hits, {len(before)} old hits")
            for rank, ((docid, score), (old_docid, old_score)) in enumerate(zip(found, before), start=1):
                if docid != old_docid or abs(score - old_score) > SCORE_TOLERANCE:
                    bad += 1
                    print(f"[verify] differs: qid {qid} rank {rank} new {docid} {score:.6f} old {old_docid} {old_score:.6f}")
    print(f"[verify] compared {len(qids)} queries x {args.hits} hits, {bad} differing lines", flush=True)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--verify", type=int, default=0, metavar="N")
    args = ap.parse_args()
    system = cfg()["systems"][args.system]
    if system["kind"] != "faiss":
        sys.exit(f"system {args.system} is not a faiss system")
    args.hits = cfg()["hits"]

    final = path("runs") / f"miracl.{args.lang}.{args.system}.txt"
    partial = final.with_name(final.name + ".partial")
    tag = f"[{args.lang} {args.system}]"
    if final.exists() and not args.verify:
        print(f"{tag} already finished: {final.name}")
        return

    topics_name = f"miracl-v1.0-{args.lang}-dev"
    topics = read_topics(topics_file(topics_name))
    all_qids = [qid for qid, _ in topics]
    queries = dict(topics)
    done = truncate_to_valid(partial, args.hits) if partial.exists() else []
    if done != all_qids[:len(done)]:
        sys.exit(f"{partial.name} does not follow the {topics_name} topic order, remove it to start over")
    todo = remaining(all_qids, done)
    print(f"{tag} {len(done)}/{len(all_qids)} queries already done", flush=True)

    if todo or args.verify:
        index = index_dir(args.lang, args.system, f"miracl-v1.0-{args.lang}{system['index_suffix']}")
        searcher = DenseSearcher(system, index)
        print(f"{tag} index and encoder loaded", flush=True)
    if args.verify:
        sys.exit(1 if verify(searcher, partial, queries, args.verify, args) else 0)

    stop = stop_on_signal()
    start = time.monotonic()
    with open(partial, "a", encoding="utf-8", newline="\n") as fh:
        for first in range(0, len(todo), args.batch_size):
            if stop.is_set():
                sys.exit(f"{tag} stopped at {len(done)}/{len(all_qids)}, run again to resume")
            batch = todo[first:first + args.batch_size]
            began = time.monotonic()
            for qid, found in searcher.search(batch, [queries[q] for q in batch], args.hits, args.threads):
                if len(found) != args.hits:
                    sys.exit(f"{tag} query {qid} returned {len(found)} hits, expected {args.hits}")
                append_block(fh, qid, found, TAG)
            done += batch
            print(f"{tag} {len(done)}/{len(all_qids)} queries, batch {time.monotonic() - began:.1f}s, "
                  f"elapsed {time.monotonic() - start:.0f}s", flush=True)

    if truncate_to_valid(partial, args.hits) != all_qids:
        sys.exit(f"{tag} {partial.name} does not hold {args.hits} hits for each of the {len(all_qids)} queries")
    os.replace(partial, final)
    print(f"{tag} finished: {final.name}", flush=True)


if __name__ == "__main__":
    main()
