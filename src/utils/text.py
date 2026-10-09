"""query and passage text for miracl from pyserini resources, cached per language."""
import json
from functools import lru_cache


@lru_cache(maxsize=None)
def _searcher(lang: str):
    from pyserini.search.lucene import LuceneSearcher
    return LuceneSearcher.from_prebuilt_index(f"miracl-v1.0-{lang}")


@lru_cache(maxsize=None)
def _topics(lang: str) -> dict[str, str]:
    from pyserini.search import get_topics
    return {str(q): v["title"] for q, v in get_topics(f"miracl-v1.0-{lang}-dev").items()}


def query_text(lang: str, qid: str) -> str:
    return _topics(lang)[qid]


def passage_text(lang: str, docid: str) -> str:
    doc = _searcher(lang).doc(docid)
    if doc is None:
        return ""
    raw = json.loads(doc.raw())
    return f"{raw.get('title', '')}\n{raw.get('text', '')}".strip()
