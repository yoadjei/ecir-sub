"""multilingual-e5-small runs for the languages in config e5_languages.

usage: PYTHONPATH=. python src/03_runs_e5.py [--langs yo sw]
passages are read from the cached pyserini lucene index, so the corpus is the same one the
bm25 and dense runs use. embeddings are cached in cache/e5 and never encoded twice.
queries are the title field of the dev topics. search is exact inner product over
normalised embeddings, which is cosine similarity.
"""
import argparse
import json
import time

import faiss
import numpy as np
import pandas as pd
import torch
from pyserini.search import get_topics
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer

from src.utils.config import cfg, path
from src.utils.io import write_run

MODEL = "intfloat/multilingual-e5-small"
BATCH = 64
# passages per saved chunk, so a crash only loses the chunk being encoded
CHUNK = 4096
THREADS = 6


def passage_text(title: str, text: str) -> str:
    body = " ".join(part for part in (title, text) if part)
    return f"passage: {body}"


def query_text(title: str) -> str:
    return f"query: {title}"


def build_run(qids: list[str], scores: np.ndarray, idx: np.ndarray, docids: list[str]) -> dict:
    """faiss rows (scores and corpus row numbers per query) to {qid: [(docid, score), ...]}."""
    return {qid: [(docids[j], float(s)) for s, j in zip(scores[i], idx[i])]
            for i, qid in enumerate(qids)}


def read_corpus(lang: str) -> tuple[list[str], list[str]]:
    """docids and prefixed passage texts, in lucene internal id order."""
    searcher = LuceneSearcher.from_prebuilt_index(f"miracl-v1.0-{lang}")
    docids, texts = [], []
    for i in range(searcher.num_docs):
        doc = json.loads(searcher.doc(i).raw())
        docids.append(str(doc["docid"]))
        texts.append(passage_text(doc["title"], doc["text"]))
    return docids, texts


def check_count(lang: str, n_passages: int) -> None:
    audit = pd.read_csv(path("results") / "data_audit.csv").set_index("lang")
    expected = int(audit.loc[lang, "corpus_passages"])
    if n_passages != expected:
        raise ValueError(f"{lang}: read {n_passages} passages, data_audit.csv has {expected}")


def encode_corpus(model: SentenceTransformer, lang: str) -> tuple[np.ndarray, list[str]]:
    cache = path("cache") / "e5"
    cache.mkdir(parents=True, exist_ok=True)
    emb_file, id_file = cache / f"{lang}.npy", cache / f"{lang}.docids.txt"
    time_file = cache / f"{lang}.enc_seconds.txt"
    if emb_file.exists() and id_file.exists():
        docids = id_file.read_text(encoding="utf-8").splitlines()
        print(f"[e5] {lang}: using cached embeddings, {len(docids)} passages, "
              f"encoding took {time_file.read_text().strip()}s")
        return np.load(emb_file), docids
    docids, texts = read_corpus(lang)
    check_count(lang, len(texts))
    parts, seconds = [], 0.0
    for start in range(0, len(texts), CHUNK):
        part_file = cache / f"{lang}.part{start // CHUNK:04d}.npz"
        if not part_file.exists():
            t0 = time.time()
            emb = model.encode(texts[start:start + CHUNK], batch_size=BATCH, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
            np.savez(part_file, emb=emb, seconds=time.time() - t0)
        part = np.load(part_file)
        parts.append(part["emb"])
        seconds += float(part["seconds"])
    emb = np.concatenate(parts)
    print(f"[e5] {lang}: encoded {len(texts)} passages in {seconds:.0f}s")
    np.save(emb_file, emb)
    time_file.write_text(f"{seconds:.0f}\n")
    id_file.write_text("\n".join(docids) + "\n", encoding="utf-8")
    for part_file in cache.glob(f"{lang}.part*.npz"):
        part_file.unlink()
    return emb, docids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=cfg()["e5_languages"])
    args = ap.parse_args()
    torch.set_num_threads(THREADS)
    model = SentenceTransformer(MODEL, device="cpu")
    for lang in args.langs:
        emb, docids = encode_corpus(model, lang)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        topics = get_topics(f"miracl-v1.0-{lang}-dev")
        qids = [str(q) for q in topics]
        qtexts = [query_text(topics[q]["title"]) for q in topics]
        qemb = model.encode(qtexts, batch_size=BATCH, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
        scores, idx = index.search(qemb, cfg()["hits"])
        out = path("runs") / f"miracl.{lang}.e5small.txt"
        write_run(build_run(qids, scores, idx, docids), out, "e5small")
        print(f"[e5] wrote {out}")


if __name__ == "__main__":
    main()
