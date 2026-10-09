"""per-language audit of miracl dev topics, qrels and corpus size.

writes results/data_audit.csv, one row per language in config.yaml.
"""
import statistics

import pandas as pd

from src.utils.config import cfg, langs, path, tier


def summarise_qrels(qrels: dict[str, dict[str, int]]) -> dict:
    per_q = [len(docs) for docs in qrels.values()]
    # pyserini 2.4.0 returns relevance labels as strings
    pos_q = [sum(1 for r in docs.values() if int(r) > 0) for docs in qrels.values()]
    return {
        "n_queries": len(qrels),
        "n_judgements": sum(per_q),
        "judgements_per_query_mean": round(statistics.mean(per_q), 3),
        "judgements_per_query_median": statistics.median(per_q),
        "positives_per_query_mean": round(statistics.mean(pos_q), 3),
        "queries_zero_positives": sum(1 for p in pos_q if p == 0),
    }


def corpus_size(lang: str) -> int:
    from pyserini.index.lucene import LuceneIndexReader
    reader = LuceneIndexReader.from_prebuilt_index(f"miracl-v1.0-{lang}")
    return int(reader.stats()["documents"])


def main() -> None:
    from pyserini.search import get_qrels, get_topics
    rows = []
    for lang in langs():
        topics = get_topics(f"miracl-v1.0-{lang}-dev")
        qrels = get_qrels(f"miracl-v1.0-{lang}-dev")
        row = {"lang": lang, "tier": tier(lang), "n_topics": len(topics)}
        row.update(summarise_qrels(qrels))
        # languages without a dense index are not measured, their lucene indexes are very large
        row["corpus_passages"] = corpus_size(lang) if cfg()["languages"][lang]["dense"] else -1
        rows.append(row)
        print(row)
    df = pd.DataFrame(rows)
    out = path("results") / "data_audit.csv"
    df.to_csv(out, index=False)
    te = df[df.lang == "te"]
    if not te.empty:
        print(f"[audit] Telugu judgements/query mean = {float(te.judgements_per_query_mean.iloc[0])}"
              f" (others: {df[df.lang != 'te'].judgements_per_query_mean.round(2).tolist()})")
    print(f"[audit] wrote {out}")


if __name__ == "__main__":
    main()
